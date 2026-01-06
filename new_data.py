#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm


import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import sample_neighbors, select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
import pickle
from utils import user_neg, get_paths
import shutil


def calculate_edge_weight(data):
    data['time'] = data['time'].astype('int64')

    min_date = data['time'].min()
    max_date = data['time'].max()
    interval = max_date - min_date

    data['recency'] = data['time'].apply(lambda x: ((x - min_date) ) / (interval))

    alpha = 3  # Decay rate
    #data['weight'] = (1 - data['event_type_prob']) * np.exp(-alpha * data['recency'])
    data['weight'] = np.exp(-alpha * data['recency'])
    
    return data

def load_dataset(data, opt):
    data = pd.read_csv(data_path, nrows=opt.max_rows)
    
    data = calculate_edge_weight(data)

    data['time'] = data.groupby('user_id')['time'].rank(method='first').astype('int64')
    
    data = data.sort_values(['user_id', 'time'], kind='mergesort')
    data['order'] = data.groupby('user_id').cumcount()

    data = data.sort_values(['item_id', 'time'], kind='mergesort')
    data['u_order'] = data.groupby('item_id').cumcount()

    data = data.reset_index(drop=True)

    return data


def generate_graph(data):
    user = data['user_id'].values
    item = data['item_id'].values
    time = data['time'].values
    weight = data['weight'].values

    graph_data = {('item','by','user'):(torch.tensor(item), torch.tensor(user)),
                  ('user','pby','item'):(torch.tensor(user), torch.tensor(item))}
    graph = dgl.heterograph(graph_data)

    graph.edges['by'].data['weight'] = torch.FloatTensor(weight)
    graph.edges['pby'].data['weight'] = torch.FloatTensor(weight)

    graph.edges['by'].data['time'] = torch.LongTensor(time)
    graph.edges['pby'].data['time'] = torch.LongTensor(time)

    graph.nodes['user'].data['user_id'] = torch.LongTensor(np.unique(user))
    graph.nodes['item'].data['item_id'] = torch.LongTensor(np.unique(item))

    return graph


def generate_user_v1_BFS(opt, user, sub_graph):
    u_temp = torch.tensor([user])
    his_user = torch.tensor([user])

    graph_i = select_topk(sub_graph, opt.item_max_length, weight='time', nodes={'user':u_temp})
    i_temp = torch.unique(graph_i.edges(etype='by')[0])
    his_item = torch.unique(graph_i.edges(etype='by')[0])
    edge_i = [graph_i.edges['by'].data[dgl.NID]]
    edge_u = []

    for _ in range(opt.rw_length-1): #k_hop
        graph_u = select_topk(sub_graph, opt.user_max_length, weight='time', nodes={'item': i_temp})  # item的邻居user
        u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-opt.user_max_length:]
        #u_temp = torch.unique(torch.cat((u_temp, graph_u.edges(etype='pby')[0])))
        graph_i = select_topk(sub_graph, opt.item_max_length, weight='time', nodes={'user': u_temp})
        his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
        #i_temp = torch.unique(torch.cat((i_temp, graph_i.edges(etype='by')[0])))
        i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
        his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
        edge_i.append(graph_i.edges['by'].data[dgl.NID])
        edge_u.append(graph_u.edges['pby'].data[dgl.NID])

    all_edge_u = torch.unique(torch.cat(edge_u))
    all_edge_i = torch.unique(torch.cat(edge_i))

    fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})

    return fin_graph



def generate_user_v2_BRW(opt, user, sub_graph):
    u_temp = torch.tensor([user])
    his_user = torch.tensor([user])
    
    graph_i = sample_neighbors(sub_graph, {'user': u_temp}, fanout=opt.rw_width, edge_dir='in', prob='weight')
    
    i_temp = torch.unique(graph_i.edges(etype='by')[0])
    his_item = torch.unique(graph_i.edges(etype='by')[0])
    
    edge_i = [graph_i.edges['by'].data[dgl.NID]]
    edge_u = []

    for _ in range(opt.rw_length-1):
        graph_u = sample_neighbors(sub_graph, {'item': i_temp}, fanout=opt.rw_width, edge_dir='in', prob='weight')
        u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user.numpy())[-opt.user_max_length:]
        
        graph_i = sample_neighbors(sub_graph, {'user': u_temp}, fanout=opt.rw_width, edge_dir='in', prob='weight')
        
        his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
        i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item.numpy())
        his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
        
        edge_i.append(graph_i.edges['by'].data[dgl.NID])
        edge_u.append(graph_u.edges['pby'].data[dgl.NID])

    all_edge_u = torch.unique(torch.cat(edge_u)) if edge_u else torch.tensor([], dtype=torch.long)
    all_edge_i = torch.unique(torch.cat(edge_i)) if edge_i else torch.tensor([], dtype=torch.long)

    fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})

    return fin_graph


def generate_user_v3_Node2Vec(opt, user, sub_graph):
    # A. Convert to Homogeneous
    g_homo = dgl.to_homogeneous(sub_graph, edata=['weight'])

    # B. Find Start Node
    user_ntype_id = sub_graph.get_ntype_id('user')
    start_node_mask = (g_homo.ndata[dgl.NTYPE] == user_ntype_id) & (g_homo.ndata[dgl.NID] == user)
    start_homo_nodes = torch.nonzero(start_node_mask, as_tuple=True)[0]

    if len(start_homo_nodes) > 0:
        # C. Perform Random Walk
        traces = dgl.sampling.node2vec_random_walk(
            g_homo, 
            start_homo_nodes, 
            p=1.0, 
            q=0.5, 
            walk_length=opt.rw_length,
            prob='weight'
        )
        
        # D. Extract Unique Nodes & Remove Padding
        visited_nodes = torch.unique(traces)
        visited_nodes = visited_nodes[visited_nodes != -1]

        # --- SAFETY NET: FIX FOR "MISSING VALIDATION" ---
        # If the walk failed (only visited the start node), force-add immediate neighbors
        if len(visited_nodes) <= 1:
            successors = g_homo.successors(start_homo_nodes)
            visited_nodes = torch.unique(torch.cat([visited_nodes, successors]))
        # -----------------------------------------------

        # E. Map Back to Heterogeneous Indices
        visited_types = g_homo.ndata[dgl.NTYPE][visited_nodes]
        visited_ids = g_homo.ndata[dgl.NID][visited_nodes]

        user_mask = (visited_types == sub_graph.get_ntype_id('user'))
        item_mask = (visited_types == sub_graph.get_ntype_id('item'))

        sel_users = visited_ids[user_mask]
        sel_items = visited_ids[item_mask]

        # F. Induce Final Subgraph
        fin_graph = dgl.node_subgraph(sub_graph, {'user': sel_users, 'item': sel_items})

        return fin_graph
    else:
        return None
    


GENERATORS = {
    '1': generate_user_v1_BFS,
    '2': generate_user_v2_BRW,
    '3': generate_user_v3_Node2Vec,
}


def generate_user(user, opt, data, graph, train_path, test_path, val_path=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0

    if opt.version not in GENERATORS:
        raise ValueError(f"Unknown version: {opt.version}")
    generate_user_version = GENERATORS[opt.version]
    
    if len(u_seq) < opt.user_min_length:
        return 0, 0
    
    # Noise Filter by edge weight
    sub_u_eid = (graph.edges['by'].data['weight'] >= opt.noise_threshold)
    sub_i_eid = (graph.edges['pby'].data['weight'] >= opt.noise_threshold)
    denoise_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)

    if denoise_graph.num_edges('by') == 0 or denoise_graph.num_edges('pby') == 0:
        return 0, 0


    for j, t  in enumerate(u_time[0:-1]):
        if j == 0:
            continue
        if j < opt.item_max_length:
            start_t = u_time[0]
        else:
            start_t = u_time[j - opt.item_max_length]

        # Temporal Slicing
        sub_u_eid = (denoise_graph.edges['by'].data['time'] < u_time[j+1]) & (denoise_graph.edges['by'].data['time'] >= start_t)
        sub_i_eid = (denoise_graph.edges['pby'].data['time'] < u_time[j+1]) & (denoise_graph.edges['pby'].data['time'] >= start_t)
        sub_graph = dgl.edge_subgraph(denoise_graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)

        #if sub_graph.num_edges('by') == 0 or sub_graph.num_edges('pby') == 0:
        #    continue
    
        # 1. Try to generate using the selected version
        fin_graph = generate_user_version(opt, user, sub_graph)

        # 2. Check for FAILURE conditions (Empty graph, missing edges, or User/Item lost)
        # We perform the check immediately to decide if we need the Fallback
        failed = False
        if fin_graph is None or fin_graph.num_edges('by') == 0 or fin_graph.num_edges('pby') == 0:
            failed = True
        else:
            # Also check if the User and Last Item still exist in the graph
            # (Random walks might sometimes walk away and "forget" the starting node)
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            if len(u_alis) == 0 or len(last_alis) == 0:
                failed = True

        # 3. FALLBACK: If V2/V3 failed, use V1 (BFS) to guarantee data consistency
        if failed:
            # Force generate using V1 so we don't lose this training/val/test sample
            fin_graph = generate_user_v1_BFS(opt, user, sub_graph)
            
            # Recalculate indices for the fallback graph
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            
            # If even V1 fails (very rare, means sub_graph was practically empty), then we must skip
            #if len(u_alis) == 0 or len(last_alis) == 0:
            #    continue

        # Clean up weights (Model doesn't need them)
        if 'weight' in fin_graph.edges['by'].data:
            del fin_graph.edges['by'].data['weight']
        if 'weight' in fin_graph.edges['pby'].data:
            del fin_graph.edges['pby'].data['weight']

        # Save Logic
        # Redefine just to be safe, though defined above
        target = u_seq[j+1]
        last_item = u_seq[j]
        u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
        last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]

        labels = {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis}

        if j < split_point-1:
            save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin', fin_graph,labels)
            train_num += 1
        if j == split_point - 1 - 1:
            save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,labels)
        if j == split_point - 1:
            save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,labels)
            test_num += 1

    return train_num, test_num


def generate_data(opt, data, graph, train_path, test_path, val_path=None):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=opt.job)(delayed(lambda u: generate_user(u, opt, data, graph, train_path, test_path, val_path))(u) for u in user)
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_min_length', type=int, default=5, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of parallel jobs')
    parser.add_argument('--rw_length', type=int, default=3, help='Depth of the random walk (formerly k_hop)')
    parser.add_argument('--rw_width', type=int, default=20, help='Branching factor')
    parser.add_argument('--noise_threshold', type=float, default=0.0, help='Minimum edge weight to consider')
    parser.add_argument('--force_graph', type=bool, default=False, help='force graph generation')
    parser.add_argument('--max_rows',type=int, default=10000, help="max dataset rows")
    parser.add_argument('--version', help='generate user version')
    opt = parser.parse_args()

    data_path, train_path, test_path, val_path, graph_path, neg_path, timestamp_path = get_paths(opt)
    
    data = load_dataset(data_path, opt)
    
    if opt.force_graph or not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    
    print('start:', datetime.datetime.now())
    
    #Debug
    #generate_user(11, opt, data, graph, train_path, test_path, val_path)


    #if train_path exists, delete the folder and contents
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)

    #"""
    all_num = generate_data(opt, data, graph, train_path, test_path, val_path)
    
    train_num = 0
    test_num = 0
    for num_ in all_num:
        train_num += num_[0]
        test_num += num_[1]
    
    print('The number of train set:', train_num)
    print('The number of test set:', test_num)

    print('Generating negative samples...')
    data_neg = user_neg(data, data['item_id'].nunique())
    f = open(neg_path, 'wb')
    pickle.dump(data_neg,f)
    f.close()

    #"""

    print('end:', datetime.datetime.now())


