#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 02:12:30 2019

@author: wtq
"""
import pandas as pd
import numpy as np
from torch.utils import data
import torch
import models
import argparse
from tqdm import tqdm
import torch.optim as optim
import sys, os
import utils
from eval_metrics import *

def load_data(dataset_str='data/'):
    df_train = pd.read_csv(dataset_str + "train.csv")
    df_test = pd.read_csv(dataset_str+ "test.csv")
    
    test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
    train_set = df_train[['user', 'item']].groupby(['user'])['item'].apply(list).reset_index()
    
    return train_set, test_set


#device =  torch.device('cuda:0')

def evaluate(test_set, model, use_gpu = 1):
    rec_list = {}
    actual_list = {}
    user_scores = {}
    # Evaluation is done for each user one at a time.
    # TODO: Optimize to evaluate batch-wise.
    item_range = torch.arange(0, model.num_items)
    if use_gpu:
            item_range = item_range.to(device)
    pred_scores = torch.mm(model.user_embeddings.weight, model.item_embeddings.weight.t())
    if model.social:
        social_scores = torch.mm(model.social_weight, pred_scores)

    for u, item_set in test_set.values:
        input_users = torch.LongTensor([u])
        if use_gpu:
            input_users = input_users.to(device)

        #user_embed = model.user_embeddings(input_users)
        scores = pred_scores[input_users].squeeze()

        if model.social:
            scores += social_scores[input_users].squeeze()
        user_scores[u] = scores.detach().cpu().numpy()
        
        _, rec_list_u = scores.topk(2000, dim=0)
        rec_list_u = rec_list_u.cpu().numpy().tolist()
        #train_list = set(train_set[train_set['user'] == u]['item'].values[0])
        #rec_list_u = [i for i in rec_list_u if i not in train_list]
        rec_list[u] = rec_list_u[:100]
        actual_list[u] = dict(zip(item_set, item_set))
        #print (actual_list[u], rec_list[u], np.isin(rec_list[u], actual_list[u].keys()))

    rows_to_output_full = ranking_metrics(rec_list, actual_list, method ='bpr')
    for row in rows_to_output_full:
        if row['Metric'] == 'Recall' and row['K'] == 50:
            return row['score']
        
train_set, test_set = load_data()
checkpoint_path = '../models/BPR_yelp_state_dict.pt'

user_neg = utils.load_negative()

model_params = {'num_users': 6858, 'num_items': 3317, 
                'embed_dim': 256, 'method': 'BPR', 
                'lambda_u': 0, 'lambda_i': 0, 
                'use_gpu': 1, 'social_dict': {},
                'user_neg': user_neg, 'user_emb': np.zeros((6858, 256)), 
                'item_emb': np.zeros((3317, 256)), 
                'social_sim': np.zeros((6858, 6858)),'social': 1}

model = models.BPR(**model_params)
#model.load_state_dict(torch.load(checkpoint_path))
model.load_state_dict(torch.load(checkpoint_path,map_location='cpu'))
#model = model.to(device)

model.eval()
print ("Evaluating on test set: ")
#evaluate(test_set, model)
evaluate(test_set, model,use_gpu = 0)