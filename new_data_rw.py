#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/11/15 7:30
# @Author : ZM7
# @File : new_data
# @Software: PyCharm

#import collections
#import collections.abc
# Comprehensive fix for Python 3.10+ compatibility with old DGL
#for name in ['Mapping', 'MutableMapping', 'Iterable', 'Sequence', 'Callable']:
#    if not hasattr(collections, name):
#        setattr(collections, name, getattr(collections.abc, name))

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


import warnings
warnings.filterwarnings('ignore')


# Calculate the relative order of the item sequence
def cal_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['order'] = range(len(data))
    return data

# Calculate the relative order of the user sequence
def cal_u_order(data):
    data = data.sort_values(['time'], kind='mergesort')
    data['u_order'] = range(len(data))
    return data

def cal_weight(data):
    data['time'] = data['time'].astype('int64')

    min_date = data['time'].min()
    max_date = data['time'].max()
    interval = max_date - min_date

    data['recency'] = data['time'].apply(lambda x: ((x - min_date) ) / (interval))

    alpha = 3  # Decay rate
    #data['weight'] = (1 - data['event_type_prob']) * np.exp(-alpha * data['recency'])
    data['weight'] = np.exp(-alpha * data['recency'])
    
    return data


def refine_time(data):
    data = data.sort_values(['time'], kind='mergesort')
    time_seq = data['time'].values
    time_gap = 1
    for i, da in enumerate(time_seq[0:-1]):
        if time_seq[i] == time_seq[i+1] or time_seq[i] > time_seq[i+1]:
            time_seq[i+1] = time_seq[i+1] + time_gap
            time_gap += 1
    data['time'] = time_seq
    return  data

def load_dataset(data, opt):
    data = pd.read_csv(data_path, nrows=opt.max_rows)
    
    data = cal_weight(data)

    data = data.groupby('user_id').apply(refine_time).reset_index(drop=True)
    data = data.groupby('user_id').apply(cal_order).reset_index(drop=True)
    data = data.groupby('item_id').apply(cal_u_order).reset_index(drop=True)

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


def generate_user(user, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop=3, val_path=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0
    # Generate training data
    if len(u_seq) < 3:
        return 0, 0
    else:
        for j, t  in enumerate(u_time[0:-1]):
            if j == 0:
                continue
            if j < item_max_length:
                start_t = u_time[0]
            else:
                start_t = u_time[j - item_max_length]

            sub_u_eid = (graph.edges['by'].data['time'] < u_time[j+1]) & (graph.edges['by'].data['time'] >= start_t)
            sub_i_eid = (graph.edges['pby'].data['time'] < u_time[j+1]) & (graph.edges['pby'].data['time'] >= start_t)
            sub_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)
            
            u_temp = torch.tensor([user])
            his_user = torch.tensor([user])

            graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user':u_temp})
            i_temp = torch.unique(graph_i.edges(etype='by')[0])
            his_item = torch.unique(graph_i.edges(etype='by')[0])
            
            edge_i = [graph_i.edges['by'].data[dgl.NID]]
            edge_u = []
            
            for _ in range(k_hop-1):
                graph_u = select_topk(sub_graph, user_max_length, weight='time', nodes={'item': i_temp})  # item的邻居user
                u_temp = np.setdiff1d(torch.unique(graph_u.edges(etype='pby')[0]), his_user)[-user_max_length:]
                #u_temp = torch.unique(torch.cat((u_temp, graph_u.edges(etype='pby')[0])))
                
                graph_i = select_topk(sub_graph, item_max_length, weight='time', nodes={'user': u_temp})
                his_user = torch.unique(torch.cat([torch.tensor(u_temp), his_user]))
                #i_temp = torch.unique(torch.cat((i_temp, graph_i.edges(etype='by')[0])))
                i_temp = np.setdiff1d(torch.unique(graph_i.edges(etype='by')[0]), his_item)
                his_item = torch.unique(torch.cat([torch.tensor(i_temp), his_item]))
                
                edge_i.append(graph_i.edges['by'].data[dgl.NID])
                edge_u.append(graph_u.edges['pby'].data[dgl.NID])

            all_edge_u = torch.unique(torch.cat(edge_u))
            all_edge_i = torch.unique(torch.cat(edge_i))
            fin_graph = dgl.edge_subgraph(sub_graph, edges={'by':all_edge_i,'pby':all_edge_u})
            target = u_seq[j+1]
            last_item = u_seq[j]
            u_alis = torch.where(fin_graph.nodes['user'].data['user_id']==user)[0]
            last_alis = torch.where(fin_graph.nodes['item'].data['item_id']==last_item)[0]
            
            # Calculate the indices of user and last_item in fin_graph separately
            if j < split_point-1:
                save_graphs(train_path+ '/' + str(user) + '/'+ str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis})
                train_num += 1
            if j == split_point - 1 - 1:
                save_graphs(val_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis': u_alis,
                             'last_alis': last_alis})
            if j == split_point - 1:
                save_graphs(test_path + '/' + str(user) + '/' + str(user) + '_' + str(j) + '.bin', fin_graph,
                            {'user': torch.tensor([user]), 'target': torch.tensor([target]), 'u_alis':u_alis, 'last_alis': last_alis})
                test_num += 1
        return train_num, test_num


def generate_data(data, graph, item_max_length, user_max_length, train_path, test_path, val_path, job=10, k_hop=3):
    user = data['user_id'].unique()
    a = Parallel(n_jobs=job)(delayed(lambda u: generate_user(u, data, graph, item_max_length, user_max_length, train_path, test_path, k_hop, val_path))(u) for u in user)
    return a


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='sample', help='data name: sample')
    parser.add_argument('--graph', action='store_true', help='no_batch')
    parser.add_argument('--item_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--user_max_length', type=int, default=50, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of epochs to train for')
    parser.add_argument('--k_hop', type=int, default=2, help='k_hop')
    parser.add_argument('--force_graph', type=bool, default=False, help='force graph generation')
    parser.add_argument('--max_rows',type=int, default=10000, help="max dataset rows")
    opt = parser.parse_args()

    data_path, train_path, test_path, val_path, graph_path, neg_path = get_paths(opt)
    
    data = load_dataset(data_path, opt)
    
    if opt.force_graph:
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    
    if not os.path.exists(graph_path):
        graph = generate_graph(data)
        save_graphs(graph_path, graph)
    else:
        graph = dgl.load_graphs(graph_path)[0][0]
    
    print('start:', datetime.datetime.now())
    
    #Debug
    generate_user(11, data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, opt.k_hop, val_path)
        

    """

    all_num = generate_data(data, graph, opt.item_max_length, opt.user_max_length, train_path, test_path, val_path, job=opt.job, k_hop=opt.k_hop)
    
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

    """

    print('end:', datetime.datetime.now())


