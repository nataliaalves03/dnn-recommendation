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
import leidenalg
import igraph as ig


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
    data = pd.read_csv(data_path) if opt.max_rows <= 0 else pd.read_csv(data_path, nrows=opt.max_rows)

    #Drop users with sequence length < user_min_length (Datasets are already filtered)
    #user_counts = data['user_id'].value_counts()
    #valid_users = user_counts[user_counts >= opt.user_min_length].index
    #data = data[data['user_id'].isin(valid_users)]
    
    #Set edge weight (context-aware)
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

    print("Max tempo:", data['time'].max())

    return data

def compute_global_communities(graph):
    # 1. Converter para Homogêneo
    hg = dgl.to_homogeneous(graph, edata=['weight'])
    
    # 2. Construir grafo igraph
    # O igraph precisa de uma lista de tuplas (src, dst) e lista de pesos
    src, dst = hg.edges()
    
    # Convertendo tensores para listas python (necessário pro igraph)
    # Nota: Usamos cpu().numpy() para garantir que não quebre se estiver em GPU (embora aqui deva ser CPU)
    edges = list(zip(src.cpu().numpy(), dst.cpu().numpy()))
    weights = hg.edata['weight'].cpu().numpy()
    
    num_nodes = hg.num_nodes()
    g_ig = ig.Graph(num_nodes, edges)
    
    # Importante: Como é um grafo de interação, tratamos como não direcionado para comunidade
    # (A conexão U->I implica uma relação mútua de comunidade)
    g_ig.to_undirected(combine_edges='first') 
    
    print(f"Running Leiden on graph with {num_nodes} nodes and {len(edges)} edges...")
    
    # 3. Rodar Leiden
    # ModularityVertexPartition é o padrão similar ao Louvain
    # weights=weights garante que interações recentes (peso maior) definam mais a comunidade
    partition = leidenalg.find_partition(
        g_ig, 
        leidenalg.ModularityVertexPartition, 
        weights=weights,
        n_iterations=-1 # Roda até convergir
    )
    
    # partition.membership dá a lista de comunidades na ordem dos nós do grafo homogêneo
    membership = np.array(partition.membership)
    num_communities = np.max(membership) + 1
    print(f"Leiden completed. Found {num_communities} mixed communities.")

    # 4. Mapear de volta para User e Item
    # dgl.to_homogeneous guarda o tipo original e o ID original
    # ntype: ID do tipo de nó (0 ou 1, dependendo da ordem interna do DGL)
    # nid: ID original dentro daquele tipo
    original_ntypes = hg.ndata[dgl.NTYPE].cpu().numpy()
    original_nids = hg.ndata[dgl.NID].cpu().numpy()
    
    # Precisamos saber qual ID numérico o DGL deu para 'user' e 'item'
    user_ntype_id = graph.get_ntype_id('user')
    item_ntype_id = graph.get_ntype_id('item')
    
    # Arrays vazios para preencher
    # Assumimos que os IDs originais são contíguos de 0 a N (o np.unique no generate_graph garante isso)
    user_comm = np.zeros(graph.num_nodes('user'), dtype=int)
    item_comm = np.zeros(graph.num_nodes('item'), dtype=int)
    
    # Preenchimento vetorizado usando máscaras numpy
    # Máscara para nós que são users
    mask_users = (original_ntypes == user_ntype_id)
    # Pega os IDs originais dos users
    u_ids = original_nids[mask_users]
    # Atribui a comunidade correspondente
    user_comm[u_ids] = membership[mask_users]
    
    # Máscara para nós que são items
    mask_items = (original_ntypes == item_ntype_id)
    i_ids = original_nids[mask_items]
    item_comm[i_ids] = membership[mask_items]
    
    return user_comm, item_comm

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

        # --- FIX: Compact IDs to range [0...N-1] ---
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
    Modified to implement Iterative/Dynamic detection.
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
        
        if opt.noise_threshold > 0:
            mask = mask & (full_w >= opt.noise_threshold)
            
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
        # Repeat the start node N times to get N walks (rw_width)
        num_walks = opt.rw_width if opt.rw_width else 10 
        seeds = start_homo_nodes.repeat(num_walks)
        
        # C. Perform Random Walk (Node2Vec)
        traces = dgl.sampling.node2vec_random_walk(
            g_homo, 
            seeds, 
            p=1.0, 
            q=0.5, 
            walk_length=opt.rw_length,
            prob='weight'
        )
        
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
    

def generate_user_v4_RW(opt, user, sub_graph):
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
    '1': generate_user_v1_BFS,
    '2': generate_user_v2_BRW,
    '3': generate_user_v3_Node2Vec,
    '4': generate_user_v4_RW,
}


def generate_user(user, opt, data, graph, train_path, test_path, val_path=None, comms_lookup=None):
    data_user = data[data['user_id'] == user].sort_values('time')
    u_time = data_user['time'].values
    u_seq = data_user['item_id'].values
    split_point = len(u_seq) - 1
    train_num = 0
    test_num = 0

    if opt.version not in GENERATORS:
        raise ValueError(f"Unknown version: {opt.version}")
    generate_user_version = GENERATORS[opt.version]
    
    if len(u_seq) < 3:
        return 0, 0
    
    """
    # Noise Filter by edge weight
    sub_u_eid = (graph.edges['by'].data['weight'] >= opt.noise_threshold)
    sub_i_eid = (graph.edges['pby'].data['weight'] >= opt.noise_threshold)
    denoise_graph = dgl.edge_subgraph(graph, edges = {'by':sub_u_eid, 'pby':sub_i_eid}, relabel_nodes=False)

    if denoise_graph.num_edges('by') == 0 or denoise_graph.num_edges('pby') == 0:
        return 0, 0
    """


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

        # --- OPTIMIZED: LOOKUP COMMUNITIES ---
        # Instead of calculating, we look up the state for the current rank 't'
        # We use 't' (u_time[j]) because that represents the current state before the target
        # OR we use 'threshold_time - 1' to represent the state at the end of the input sequence.
        
        # Using u_time[j] (the time of the last item in input) is safer.
        current_rank = u_time[j]
        
        if current_rank in comms_lookup:
            current_u_comm, current_i_comm = comms_lookup[current_rank]
            sub_graph.nodes['user'].data['community_id'] = current_u_comm
            sub_graph.nodes['item'].data['community_id'] = current_i_comm
        else:
            # Fallback (should not happen if pre-computed correctly)
            sub_graph.nodes['user'].data['community_id'] = torch.zeros(sub_graph.num_nodes('user'), dtype=torch.long)
            sub_graph.nodes['item'].data['community_id'] = torch.zeros(sub_graph.num_nodes('item'), dtype=torch.long)
        # -------------------------------------
        

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
    parser.add_argument('--user_min_length', type=int, default=5, help='most recent')
    parser.add_argument('--job', type=int, default=10, help='number of parallel jobs')
    parser.add_argument('--rw_length', type=int, default=3, help='Depth of the random walk (formerly k_hop)')
    parser.add_argument('--rw_width', type=int, default=20, help='Branching factor')
    parser.add_argument('--noise_threshold', type=float, default=0.0, help='Minimum edge weight to consider')
    parser.add_argument('--force_graph', type=bool, default=False, help='force graph generation')
    parser.add_argument('--max_rows',type=int, default=0, help="max dataset rows (0 is disabled)")
    parser.add_argument('--version', help='generate user version')
    opt = parser.parse_args()

    data_path, train_path, test_path, val_path, graph_path, neg_path, timestamp_path = get_paths(opt)
    
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

    #if train_path exists, delete the folder and contents
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    if os.path.exists(val_path):
        shutil.rmtree(val_path)

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


