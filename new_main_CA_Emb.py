#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/01/10 10:00
# @Author : ZM7, nataliaalves03
# @File : new_main
# @Software: PyCharm

import datetime
import torch
from sys import exit
import pandas as pd
import numpy as np
from DGSRCA_Emb import DGSRCA, collate, collate_test
from dgl import load_graphs
import pickle
from utils import myFloder, get_paths
import warnings
import argparse
import os
import sys
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from DGSR_utils import eval_metric, mkdir_if_not_exist, Logger



print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--data', default='sample', help='data name: sample')
parser.add_argument('--batchSize', type=int, default=50, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=50, help='hidden state size')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--l2', type=float, default=0.0001, help='l2 penalty')
parser.add_argument('--user_update', default='rnn')
parser.add_argument('--item_update', default='rnn')
parser.add_argument('--user_long', default='orgat')
parser.add_argument('--item_long', default='orgat')
parser.add_argument('--user_short', default='att')
parser.add_argument('--item_short', default='att')
parser.add_argument('--feat_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--attn_drop', type=float, default=0.3, help='drop_out')
parser.add_argument('--layer_num', type=int, default=3, help='GNN layer')
parser.add_argument('--item_max_length', type=int, default=50, help='the max length of item sequence')
parser.add_argument('--user_max_length', type=int, default=50, help='the max length of use sequence')

#not in use, only filename
parser.add_argument('--k_hop', type=int, default=3, help='k hop for subgraph extraction')
parser.add_argument('--rw_length', type=int, default=3, help='Depth of the random walk (formerly k_hop)')
parser.add_argument('--rw_width', type=int, default=20, help='Branching factor: max neighbors sampled per node (formerly fanout)')
parser.add_argument('--version', type=str, help='data version')

parser.add_argument('--gpu', default='0')
parser.add_argument('--last_item', action='store_true', help='aggreate last item')
parser.add_argument("--record", action='store_true', default=False, help='record experimental results')
parser.add_argument("--val", action='store_true', default=False)
parser.add_argument("--model_record", action='store_true', default=False, help='record model')

opt = parser.parse_args()
args, extras = parser.parse_known_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
device = torch.device('cuda:0')

print('Parameters:')
print(opt)

data_path, train_path, test_path, val_path, graph_path, neg_path, timestamp_path = get_paths(opt)

print('Train data:', train_path)
print('Graph data:', graph_path)

if opt.record:
    log_file = f'results/{timestamp_path}'
    mkdir_if_not_exist(log_file)
    sys.stdout = Logger(log_file)
    print(f'Logging to {log_file}')
if opt.model_record:
    model_file = f'models/{timestamp_path}'


# Load graph and get user/item/community numbers
g_list, g_labels = load_graphs(graph_path)
full_graph = g_list[0]
user_num = full_graph.num_nodes('user')
item_num = full_graph.num_nodes('item')
community_num = None
if 'max_community_id' in g_labels:
    # +1 porque IDs vão de 1 a Max, então precisamos de espaço para o 0 (padding/isolado)
    community_num = g_labels['max_community_id'].item() + 1
    print(f"Community info loaded from metadata: {community_num} communities.")
else:
    print("Warning: No community info found in graph. Running without community embeddings.")


train_set = myFloder(train_path, load_graphs)
test_set = myFloder(test_path, load_graphs)
if opt.val:
    val_set = myFloder(val_path, load_graphs)

print('train number:', train_set.size)
print('test number:', test_set.size)
print('user number:', user_num)
print('item number:', item_num)

f = open(neg_path, 'rb')
data_neg = pickle.load(f) # Used for evaluating the test set
train_data = DataLoader(dataset=train_set, batch_size=opt.batchSize, collate_fn=collate, shuffle=True, pin_memory=True, num_workers=12)
test_data = DataLoader(dataset=test_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=0)
if opt.val:
    val_data = DataLoader(dataset=val_set, batch_size=opt.batchSize, collate_fn=lambda x: collate_test(x, data_neg), pin_memory=True, num_workers=2)

# Initialize the model
model = DGSRCA(user_num=user_num, item_num=item_num, input_dim=opt.hidden_size, item_max_length=opt.item_max_length,
             user_max_length=opt.user_max_length, feat_drop=opt.feat_drop, attn_drop=opt.attn_drop, user_long=opt.user_long, user_short=opt.user_short,
             item_long=opt.item_long, item_short=opt.item_short, user_update=opt.user_update, item_update=opt.item_update, last_item=opt.last_item,
             layer_num=opt.layer_num, community_num=community_num).cuda()
optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
loss_func = nn.CrossEntropyLoss()

# Best result storage: Hit@5, Hit@10, Hit@20, NDCG@5, NDCG@10, NDCG@20, MRR@5, MRR@10, MRR@20
best_result = [0, 0, 0, 0, 0, 0, 0, 0, 0]
best_epoch = [0, 0, 0, 0, 0, 0, 0, 0, 0]
stop_num = 0

for epoch in range(opt.epoch):
    stop = True
    epoch_loss = 0
    iter = 0
    print('start training: ', datetime.datetime.now())
    model.train()
    for user, batch_graph, label, last_item in train_data:
        iter += 1
        score = model(batch_graph.to(device), user.to(device), last_item.to(device), is_training=True)
        loss = loss_func(score, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if iter % 400 == 0:
            print('Iter {}, loss {:.4f}'.format(iter, epoch_loss/iter), datetime.datetime.now())
    epoch_loss /= iter
    model.eval()
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss), '=============================================')

    # val
    if opt.val:
        print('start validation: ', datetime.datetime.now())
        val_loss_all, top_val = [], []
        with torch.no_grad:
            for user, batch_graph, label, last_item, neg_tar in val_data:
                score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar], -1).to(device), is_training=False)
                val_loss = loss_func(score, label.cuda())
                val_loss_all.append(val_loss.append(val_loss.item()))
                top_val.append(top.detach().cpu().numpy())

            metrics = eval_metric(top_val)
            print(f"Validation Loss:{np.mean(val_loss_all):.4f}")
            print('Validation \tHit@5:%.4f\tHit@10:%.4f\tHit@20:%.4f\tNDCG@5:%.4f\tNDCG@10:%.4f\tNDCG@20:%.4f\tMRR@5:%.4f\tMRR@10:%.4f\tMRR@20:%.4f' %
              (metrics['Hit@5'], metrics['Hit@10'], metrics['Hit@20'], metrics['NDCG@5'], metrics['NDCG@10'], metrics['NDCG@20'], metrics['MRR@5'], metrics['MRR@10'], metrics['MRR@20']))

    # test
    print('start predicting: ', datetime.datetime.now())
    all_top, all_label, all_loss = [], [], []
    iter = 0
    with torch.no_grad():
        for user, batch_graph, label, last_item, neg_tar in test_data:
            iter+=1
            score, top = model(batch_graph.to(device), user.to(device), last_item.to(device), neg_tar=torch.cat([label.unsqueeze(1), neg_tar],-1).to(device),  is_training=False)
            test_loss = loss_func(score, label.cuda())
            all_loss.append(test_loss.item())
            all_top.append(top.detach().cpu().numpy())
            all_label.append(label.numpy())
            if iter % 200 == 0:
                print('Iter {}, test_loss {:.4f}'.format(iter, np.mean(all_loss)), datetime.datetime.now())

        # Calculate metrics
        metrics = eval_metric(all_top)
        
        # Update best results (using Hit/Recall and NDCG for stopping criteria/saving)
        # Note: Recall is equal to Hit in this context.
        if metrics['Hit@5'] > best_result[0]:
            best_result[0] = metrics['Hit@5']
            best_epoch[0] = epoch
            stop = False
        if metrics['Hit@10'] > best_result[1]:
            if opt.model_record:
                torch.save(model.state_dict(), 'save_models/' + model_file + '.pkl')
            best_result[1] = metrics['Hit@10']
            best_epoch[1] = epoch
            stop = False
        if metrics['Hit@20'] > best_result[2]:
            best_result[2] = metrics['Hit@20']
            best_epoch[2] = epoch
            stop = False
        
        # NDCG
        if metrics['NDCG@5'] > best_result[3]:
            best_result[3] = metrics['NDCG@5']
            best_epoch[3] = epoch
            stop = False
        if metrics['NDCG@10'] > best_result[4]:
            best_result[4] = metrics['NDCG@10']
            best_epoch[4] = epoch
            stop = False
        if metrics['NDCG@20'] > best_result[5]:
            best_result[5] = metrics['NDCG@20']
            best_epoch[5] = epoch
            stop = False

        # MRR
        if metrics['MRR@5'] > best_result[6]:
            best_result[6] = metrics['MRR@5']
            best_epoch[6] = epoch
            stop = False
        if metrics['MRR@10'] > best_result[7]:
            best_result[7] = metrics['MRR@10']
            best_epoch[7] = epoch
            stop = False
        if metrics['MRR@20'] > best_result[8]:
            best_result[8] = metrics['MRR@20']
            best_epoch[8] = epoch
            stop = False

        if stop:
            stop_num += 1
        else:
            stop_num = 0
        
        print(f"Test Loss: {np.mean(all_loss):.4f}")
        
        metrics = {k: np.round(v, 4) for k, v in metrics.items()}
        best_result = np.round(best_result, 4).tolist()

        print('Current Epoch \tHit@5:%.4f\tHit@10:%.4f\tHit@20:%.4f\tNDCG@5:%.4f\tNDCG@10:%.4f\tNDCG@20:%.4f\tMRR@5:%.4f\tMRR@10:%.4f\tMRR@20:%.4f' %
              (metrics['Hit@5'], metrics['Hit@10'], metrics['Hit@20'], metrics['NDCG@5'], metrics['NDCG@10'], metrics['NDCG@20'], metrics['MRR@5'], metrics['MRR@10'], metrics['MRR@20']))
        
        print('Best Results \tHit@5:%.4f\tHit@10:%.4f\tHit@20:%.4f\tNDCG@5:%.4f\tNDCG@10:%.4f\tNDCG@20:%.4f\tMRR@5:%.4f\tMRR@10:%.4f\tMRR@20:%.4f' %
              (best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_result[6], best_result[7], best_result[8]))
        
        print(f"Best Epochs: {best_epoch}")