#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026/01/10 10:00
# @Author : ZM7, nataliaalves03
# @File : DGSR_utils
# @Software: PyCharm

import numpy as np
import sys
import os

def eval_metric(all_top, k_list=[5, 10, 20]):
    """
    Evaluates predictions using Hit, Recall, NDCG, and MRR metrics.
    Vectorized implementation for speed.

    Args:
        all_top: List of numpy arrays (batch_size, num_candidates).
                 It assumes the target item (ground truth) is always at index 0,
                 and the rest are negative samples.
        k_list: List of K values for @K metrics.

    Returns:
        dict: A dictionary containing the mean value for each metric @ K.
    """
    # Concatenate all batches into a single matrix
    predictions = np.concatenate(all_top, axis=0)

    # The target item is always at index 0. We need to find the rank of index 0.
    # argsort(-preds) sorts indices by score descending.
    # argsort().argsort() retrieves the rank (0-based index in the sorted list).
    # We take [:, 0] to get the rank of the target item.
    ranks = (-predictions).argsort(axis=1).argsort(axis=1)[:, 0]

    results = {}

    for k in k_list:
        # Hit Rate @ K
        # 1 if rank < k, else 0
        hits = (ranks < k).astype(np.float32)
        results[f'Hit@{k}'] = np.mean(hits)

        # Recall @ K
        # In single-target next-item recommendation, Recall@K is mathematically identical to Hit@K.
        results[f'Recall@{k}'] = results[f'Hit@{k}']

        # MRR @ K (Mean Reciprocal Rank)
        # 1 / (rank + 1) if rank < k, else 0
        mrr_vals = 1.0 / (ranks + 1)
        mrr_vals[ranks >= k] = 0.0
        results[f'MRR@{k}'] = np.mean(mrr_vals)

        # NDCG @ K (Normalized Discounted Cumulative Gain)
        # 1 / log2(rank + 2) if rank < k, else 0
        ndcg_vals = 1.0 / np.log2(ranks + 2)
        ndcg_vals[ranks >= k] = 0.0
        results[f'NDCG@{k}'] = np.mean(ndcg_vals)

    return results


def mkdir_if_not_exist(file_name):
    dir_name = os.path.dirname(file_name)
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


class Logger(object):
    """
    This class enables printing output to both the console and a file simultaneously
    without changing the original print code significantly.
    Usage: simply add `sys.stdout = Logger(log_file_path)` in the program.
    """
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        pass