#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/01/10 10:00
# @Author : ZM7, nataliaalves03
# @File : new_data
# @Software: PyCharm


import dgl
import pandas as pd
import numpy as np
import datetime
import argparse
from dgl.sampling import select_topk
import torch
import os
from dgl import save_graphs
from joblib import Parallel, delayed
import pickle
from utils import user_neg, get_paths
import shutil
import leidenalg
import igraph as ig


def calculate_edge_weight(data, alpha=3, func='log'):
    """Assign edge weights using a logarithmic function of recency.

    Recency is normalized to [0,1] (older -> 0, more recent -> 1).
    Weight = log1p(alpha * recency) / log1p(alpha) so weights are in [0,1]
    and increase with recency (more recent -> higher weight).
    """
    data['time'] = data['time'].astype('int64')

    if opt.version == 1:
        data['weight'] = data['time']
        return data

    min_date = data['time'].min()
    max_date = data['time'].max()
    interval = max_date - min_date

    if interval == 0:
        data['weight'] = data['time']
    else:
        #Recency is normalized to [0,1] (older -> 0, more recent -> 1).
        data['recency'] = (data['time'] - min_date) / interval

        if func == 'log':
            # Normalized logarithmic scaling: more recent => larger weight in [0,1]
            data['weight'] = np.log1p(alpha * data['recency']) / np.log1p(alpha)
        elif func == 'exp':
            # Exponential decay (older -> smaller weight)
            data['weight'] = np.exp(-alpha * data['recency'])

    return data


def load_dataset(data, opt):
    data = pd.read_csv(data_path) # if opt.n_users <= 0 else pd.read_csv(data_path, nrows=opt.n_users)

    #Drop users with less than 5 interactions
    MIN_INTERACTIONS = 5
    user_counts = data['user_id'].value_counts()
    valid_users = user_counts[user_counts >= MIN_INTERACTIONS].index
    data = data[data['user_id'].isin(valid_users)].reset_index(drop=True)

    # Sample users if n_users is specified
    if opt.n_users > 0:
        data = data[data['user_id'] < opt.n_users]
        print(f"Sampled {opt.n_users} users")
    
    #rename item_id to be continuous
    unique_items = data['item_id'].unique()
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items)}
    data['item_id'] = data['item_id'].map(item_mapping)
    #rename user_id to be continuous
    unique_users = data['user_id'].unique()
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
    data['user_id'] = data['user_id'].map(user_mapping)

    
    #Set edge weight
    data = calculate_edge_weight(data)

    #Rank time for each user
    data['time'] = data.groupby('user_id')['time'].rank(method='first').astype('int64')
    
    #Order user history
    data = data.sort_values(['user_id', 'time'], kind='mergesort')
    data['order'] = data.groupby('user_id').cumcount()

    #Order item history
    data = data.sort_values(['item_id', 'time'], kind='mergesort')
    data['u_order'] = data.groupby('item_id').cumcount()   

    data = data.reset_index(drop=True)

    print("Max time:", data['time'].max())
    print("# Users:", data['user_id'].max()+1)
    print("# Items:", data['item_id'].max()+1)
    print("# Interactions:", len(data))
    
    return data


def compute_community_for_slice(users, items, weights, num_users, num_items, 
                                prev_u_comm=None, prev_i_comm=None):
    """
    Updated to support Warm Start (Dynamic) detection with ID remapping.
    Fixes the 'Value cannot exceed length of list' error by compacting community IDs.
    """
    # 1. Identify active nodes
    active_u_unique = np.unique(users)
    active_i_unique = np.unique(items)
    
    n_active_u = len(active_u_unique)
    n_active_i = len(active_i_unique)
    
    # 2. Virtual Mapping (User -> [0..Nu], Item -> [Nu..Nu+Ni])
    u_map = {uid: i for i, uid in enumerate(active_u_unique)}
    i_map = {iid: i + n_active_u for i, iid in enumerate(active_i_unique)}
    
    # 3. Build Edges for igraph
    ig_edges = [(u_map[u], i_map[i]) for u, i in zip(users, items)]
    
    # 4. Construct Graph
    g_ig = ig.Graph(n=n_active_u + n_active_i, edges=ig_edges)
    g_ig.to_undirected(combine_edges='first')
    
    # 5. Prepare Initial Membership (Warm Start)
    initial_membership = None
    
    if prev_u_comm is not None and prev_i_comm is not None:
        # Initialize with -1
        initial_membership = [-1] * (n_active_u + n_active_i)
        
        # Map Users from Global State
        for uid in active_u_unique:
            if prev_u_comm[uid] >= 0:
                initial_membership[u_map[uid]] = int(prev_u_comm[uid])
                
        # Map Items from Global State
        for iid in active_i_unique:
            if prev_i_comm[iid] >= 0:
                initial_membership[i_map[iid]] = int(prev_i_comm[iid])

        # --- Compact IDs to range [0...N-1] ---
        # Leiden requires community IDs to be < Number of Nodes.
        # We map the sparse Global IDs to a compact contiguous range for this specific slice.
        
        # Get all unique valid community IDs present in this slice
        unique_ids = sorted(list(set(x for x in initial_membership if x != -1)))
        
        # Create a mapping: Old_ID -> New_Compact_ID (0, 1, 2, ...)
        id_remapping = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}
        
        # Assign new IDs
        # Any node with -1 (new/unassigned) gets a unique ID continuing from where we left off
        next_id = len(unique_ids)
        
        for i in range(len(initial_membership)):
            val = initial_membership[i]
            if val != -1:
                initial_membership[i] = id_remapping[val]
            else:
                initial_membership[i] = next_id
                next_id += 1
        # -------------------------------------------

    # 6. Run Leiden
    if initial_membership:
        partition = leidenalg.find_partition(
            g_ig, 
            leidenalg.ModularityVertexPartition, 
            weights=weights,
            initial_membership=initial_membership,
            n_iterations=-1
        )
    else:
        partition = leidenalg.find_partition(
            g_ig, 
            leidenalg.ModularityVertexPartition, 
            weights=weights,
            n_iterations=-1
        )
        
    membership = partition.membership
    
    # 7. Map back to Global IDs and update tensors
    u_tensor = torch.zeros(num_users, dtype=torch.long)
    i_tensor = torch.zeros(num_items, dtype=torch.long)
    
    # Map Users
    for virt_id in range(n_active_u):
        orig_uid = active_u_unique[virt_id]
        u_tensor[orig_uid] = membership[virt_id]
        
    # Map Items
    for virt_id in range(n_active_u, len(membership)):
        orig_iid = active_i_unique[virt_id - n_active_u]
        i_tensor[orig_iid] = membership[virt_id]
        
    return u_tensor, i_tensor


def precompute_rank_communities(data, opt):
    """
    Implement Iterative/Dynamic community detection.
    We maintain the state of communities and pass it to the next step.
    """
    print("Pre-computing Dynamic communities (Warm Start)...")
    
    unique_times = np.sort(data['time'].unique())
    
    full_u = data['user_id'].values
    full_i = data['item_id'].values
    full_t = data['time'].values
    full_w = data['weight'].values
    
    n_users = data['user_id'].max() + 1
    n_items = data['item_id'].max() + 1
    
    comms_lookup = {}
    global_max = 0
    
    # Initialize global community state with -1 (meaning unassigned)
    # Using numpy arrays for faster indexing
    global_u_comm = np.full(n_users, -1, dtype=int)
    global_i_comm = np.full(n_items, -1, dtype=int)
    
    for t in unique_times:
        mask = (full_t <= t)
            
        if not np.any(mask):
            comms_lookup[t] = (torch.zeros(n_users, dtype=torch.long), 
                               torch.zeros(n_items, dtype=torch.long))
            continue
            
        s_users = full_u[mask]
        s_items = full_i[mask]
        s_weights = full_w[mask]
        
        if len(s_users) == 0:
            comms_lookup[t] = (torch.zeros(n_users, dtype=torch.long), 
                               torch.zeros(n_items, dtype=torch.long))
            continue

        # --- DYNAMIC UPDATE ---
        # We pass the CURRENT global state as the PREVIOUS state for this slice
        # The function will use this to initialize Leiden
        u_tensor, i_tensor = compute_community_for_slice(
            s_users, s_items, s_weights, n_users, n_items,
            prev_u_comm=global_u_comm,
            prev_i_comm=global_i_comm
        )
        
        # Save results for graph generation
        comms_lookup[t] = (u_tensor, i_tensor)

        # Update global state for the NEXT iteration
        # Note: We update using numpy(). Since tensor indices match global IDs, this is direct.
        # Any node not in the current slice retains its old community (or -1) 
        # But 'u_tensor' has 0s for missing nodes. We need to be careful not to overwrite 
        # existing communities with 0s if the node is temporarily inactive?
        # Actually, since mask is cumulative (full_t <= t), nodes never disappear, they only appear.
        # So we can safely overwrite.
        global_u_comm = u_tensor.numpy()
        global_i_comm = i_tensor.numpy()

        current_max = max(u_tensor.max().item(), i_tensor.max().item())
        if current_max > global_max:
            global_max = current_max
            
        if t % 50 == 0:
            print(f"Processed Rank {t}/{unique_times[-1]}")
        
    print(f"Pre-computation finished. {len(comms_lookup)} states cached. Max Community ID: {global_max}")
    return comms_lookup, global_max


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

    unique_users = np.unique(user)
    unique_items = np.unique(item)

    graph.nodes['user'].data['user_id'] = torch.LongTensor(unique_users)
    graph.nodes['item'].data['item_id'] = torch.LongTensor(unique_items)
    
    graph.nodes['user'].data['community_id'] = torch.zeros(len(unique_users), dtype=torch.long)
    graph.nodes['item'].data['community_id'] = torch.zeros(len(unique_items), dtype=torch.long)

    return graph


def generate_user_greedy(opt, user, sub_graph):
    u_temp = torch.tensor([user])
    his_user = torch.tensor([user])

    graph_i = select_topk(sub_graph, opt.item_max_length, weight='time', nodes={'user':u_temp})
    i_temp = torch.unique(graph_i.edges(etype='by')[0])
    his_item = torch.unique(graph_i.edges(etype='by')[0])
    edge_i = [graph_i.edges['by'].data[dgl.NID]]
    edge_u = []

    for _ in range(opt.k_hop-1):
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

def generate_user_RW(opt, user, sub_graph):
    # A. Convert to Homogeneous
    g_homo = dgl.to_homogeneous(sub_graph, edata=['weight'])

    # B. Find Start Node
    user_ntype_id = sub_graph.get_ntype_id('user')
    start_node_mask = (g_homo.ndata[dgl.NTYPE] == user_ntype_id) & (g_homo.ndata[dgl.NID] == user)
    start_homo_nodes = torch.nonzero(start_node_mask, as_tuple=True)[0]

    if len(start_homo_nodes) > 0:
        
        num_walks = opt.rw_width if opt.rw_width else 10 
        seeds = start_homo_nodes.repeat(num_walks)

        traces, _ = dgl.sampling.random_walk(g_homo, seeds, length=opt.rw_length, prob='weight')

        # D. Extract Unique Nodes (Flatten all walks into one set)
        visited_nodes = torch.unique(traces)
        visited_nodes = visited_nodes[visited_nodes != -1]

        # Safety Net (Force Neighbors if empty)
        if len(visited_nodes) <= 1:
            successors = g_homo.successors(start_homo_nodes)
            visited_nodes = torch.unique(torch.cat([visited_nodes, successors]))

        # E. Map Back and F. Induce Graph (Same as before)
        visited_types = g_homo.ndata[dgl.NTYPE][visited_nodes]
        visited_ids = g_homo.ndata[dgl.NID][visited_nodes]
        
        user_mask = (visited_types == sub_graph.get_ntype_id('user'))
        item_mask = (visited_types == sub_graph.get_ntype_id('item'))
        
        sel_users = visited_ids[user_mask]
        sel_items = visited_ids[item_mask]
        
        fin_graph = dgl.node_subgraph(sub_graph, {'user': sel_users, 'item': sel_items})

        return fin_graph
    else:
        return None


GENERATORS = {
    '1': generate_user_greedy,
    '2': generate_user_RW,
}


def generate_user(user, opt, data, graph, train_path, test_path, val_path=None, comms_lookup=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    count_fallback = 0

    if opt.version not in GENERATORS:
        raise ValueError(f"Unknown version: {opt.version}")
    generate_user_version = GENERATORS[opt.version]
    
    if len(u_seq) < 3:
        return 0, 0
    

    for j, t  in enumerate(u_time[0:-1]):
        if j == 0:
            continue
        if j < opt.item_max_length:
            start_t = u_time[0]
        else:
            start_t = u_time[j - opt.item_max_length]

        # Temporal Slicing
        sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
        sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
        sub_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)

        # LOOKUP COMMUNITIES
        # Instead of calculating, we look up the state for the current rank 't'
        # We use 't' (u_time[j]) because that represents the current state before the target
        current_rank = u_time[j]
        
        if current_rank in comms_lookup:
            current_u_comm, current_i_comm = comms_lookup[current_rank]
            sub_graph.nodes['user'].data['community_id'] = current_u_comm
            sub_graph.nodes['item'].data['community_id'] = current_i_comm

    
        # 1. Try to generate using the selected version
        fin_graph = generate_user_version(opt, user, sub_graph)

        # 2. Check for FAILURE conditions (Empty graph, missing edges, or User/Item lost)
        if fin_graph is None or fin_graph.num_edges('by') == 0 or fin_graph.num_edges('pby') == 0:
            fin_graph = generate_user_greedy(opt, user, sub_graph)
            count_fallback += 1
 
        # Clean up weights (Model doesn't need them)
        if 'weight' in fin_graph.edges['by'].data:
            del fin_graph.edges['by'].data['weight']
        if 'weight' in fin_graph.edges['pby'].data:
            del fin_graph.edges['pby'].data['weight']

        # Save Logic
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

    if count_fallback > 0:
        print(f"User {user}: Fallbacks used: {count_fallback} out of {len(u_seq)-2} samples.")

    return train_num, test_num


def generate_data(opt, data, graph, train_path, test_path, val_path=None,comms_lookup=None):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=opt.job)(delayed(lambda u: generate_user(u, opt, data, graph, train_path, test_path, val_path, comms_lookup))(u) for u in user)
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of parallel jobs')
    parser.add_argument('--k_hop', type=int, default=3, help='k hop for subgraph extraction')
    parser.add_argument('--rw_length', type=int, default=10, help='Depth of the random walk')
    parser.add_argument('--rw_width', type=int, default=10, help='Branching factor')
    parser.add_argument('--force_graph', type=bool, default=False, help='force graph generation')
    parser.add_argument('--n_users',type=int, default=0, help="max dataset users (0 is disabled)")
    parser.add_argument('--version', help='generate user version')
    opt = parser.parse_args()

    data_path, train_path, test_path, val_path, graph_path, neg_path, timestamp_path = get_paths(opt)

    #if train_path exists, delete the folder and contents
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    
    data = load_dataset(data_path, opt)

    print('start comm:', datetime.datetime.now())
    comms_lookup, max_comm_id = precompute_rank_communities(data, opt)
    print('end comm:', datetime.datetime.now())
    
    if opt.force_graph or not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, [graph], {'max_community_id': torch.tensor([max_comm_id])})
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    
    print('start:', datetime.datetime.now())

    
    #Debug
    #generate_user(11, opt, data, graph, train_path, test_path, val_path, comms_lookup)


    #"""
    all_num = generate_data(opt, data, graph, train_path, test_path, val_path, comms_lookup)
    
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


