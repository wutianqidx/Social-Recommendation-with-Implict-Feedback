import models
import argparse
from tqdm import tqdm
import torch.optim as optim
import networkx as nx
import torch
from torch.utils import data
import sys, os
import numpy as np
import utils
from eval_metrics import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Evaluation to get Recall@50 and NDCG@50
def evaluate(test_set, train_set, model, mode = 'test', use_gpu = 0, random_split=1):
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

        scores = pred_scores[input_users].squeeze()

        if model.social:
            scores += social_scores[input_users].squeeze()
        user_scores[u] = scores.detach().cpu().numpy()
        
        _, rec_list_u = scores.topk(2000, dim=0)
        rec_list_u = rec_list_u.cpu().numpy().tolist()
        if random_split:
            train_list = set(train_set[train_set['user'] == u]['item'].values[0])
            rec_list_u = [i for i in rec_list_u if i not in train_list]
        rec_list[u] = rec_list_u[:100]
        actual_list[u] = dict(zip(item_set, item_set))
        #print (actual_list[u], rec_list[u], np.isin(rec_list[u], actual_list[u].keys()))

    rows_to_output_full = ranking_metrics(rec_list, actual_list, method ='bpr')
    #print ("\n\n")
    for row in rows_to_output_full:
        if row['Metric'] == 'Recall' and row['K'] == 50:
            return row['score']

# Parameters Setting
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run script parameters')

    parser.add_argument('--dataset', type=str, nargs='?', default='yelp',
                        help='Dataset Name')

    parser.add_argument('--method', type=str, nargs='?', default='BPR',
                        help='method name')

    parser.add_argument('--random_split', type=int, nargs='?', default=0,
                        help='argument to specify random split')

    parser.add_argument('--embed_dim', type=int, nargs='?', default=256,
                        help='embedding dimension')

    parser.add_argument('--batch_size', type=int, nargs='?', default=256,
                        help='batch size')
    
    parser.add_argument('--social', type=int, nargs='?', default=1,
                        help='Flag to use social information')

    parser.add_argument('--epochs', type=int, nargs='?', default=50,
                        help='# epochs')

    parser.add_argument('--learning_rate', type=float, nargs='?', default=0.0001,
                        help='learning rate')

    parser.add_argument('--weight_decay', type=float, nargs='?', default=0.0001,
                        help='L2 reg weight decay')

    parser.add_argument('--lambda_u', type=float, nargs='?', default=0.0000,
                        help='L2 reg hyper-param (users)')

    parser.add_argument('--lambda_i', type=float, nargs='?', default=0.0000,
                        help='L2 reg hyper-param (items)')

    parser.add_argument('--val_iter', type=int, nargs='?', default=1,
                        help='Interval for validation')

    parser.add_argument('--use_gpu', type=int, nargs='?', default=0,
                        help='Flag to use GPU')
    
    parser.add_argument('--new_model', type=int, nargs='?', default=1,
                        help='Flag to train new model')

    args = parser.parse_args()
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load Data
    train_ratings, train_set, test_set, num_items, num_users = utils.load_data(args.dataset, 
                                                                               args.random_split)
    social_dict = utils.load_social()
    item_cats = utils.load_category()
    user_neg = utils.load_negative()
    user_emb, item_emb, social_sim = utils.load_embedding(embed_dim=args.embed_dim, 
                                                          random_split=args.random_split)
    
    train_users = [x[0] for x in train_ratings]
    train_users = {x:0 for x in train_users}

    # data loader params
    train_loader_params = {'batch_size': args.batch_size,
                           'shuffle': True,
                           'num_workers': 1}

    test_loader_params = {'batch_size': args.batch_size,
                          'shuffle': False,
                          'num_workers': 1}

    training_set = utils.Dataset(train_ratings)
    train_generator = data.DataLoader(training_set, **train_loader_params)

    model_params = {'num_users': num_users, 'num_items': num_items, 
                    'embed_dim': args.embed_dim, 'method': args.method, 
                    'lambda_u': args.lambda_u, 'lambda_i': args.lambda_i, 
                    'use_gpu': args.use_gpu, 'social_dict': social_dict,
                    'user_neg': user_neg, 'user_emb': user_emb, 
                    'item_emb': item_emb, 'social_sim': social_sim,
                    'social': args.social}

    checkpoint_path = '../models/{}_{}.pt'.format(args.method, args.dataset)
    model = models.BPR(**model_params)
    if not args.new_model:
        model.load_state_dict(torch.load(checkpoint_path))
    if args.use_gpu:
        model = model.to(device)

    if args.method == 'BPR':
       optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Start Training
    best_val_performance = 0.0
    for epoch in range(args.epochs):
        if epoch == 15:
            optimizer.param_groups[0]['lr'] = 0.00001
        if epoch == 35:
            optimizer.param_groups[0]['weight_decay'] = 0.0000

        print(optimizer.param_groups[0]['lr'])
        model.train()
        epoch_loss = 0.
        for user_batch, item_batch in tqdm(train_generator):
            model.zero_grad()
            optimizer.zero_grad()
            user_batch = torch.LongTensor(user_batch)
            item_batch = torch.LongTensor(item_batch)
            if args.use_gpu:
                user_batch = user_batch.to(device)
                item_batch = item_batch.to(device)
            pos_preds = model(user_batch, item_batch)
            negative_items = model.sample_negs(item_batch, 1) # [B, 1]
            if args.use_gpu:
                negative_items = negative_items.to(device)
            neg_preds = model(user_batch, negative_items.view(-1))
            batch_loss = model.loss(pos_preds, neg_preds)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss
        epoch_loss /= len(train_generator)

        if epoch % args.val_iter == 0:
            model.eval()
            val_performance = evaluate(test_set, train_set, model, use_gpu=args.use_gpu, 
                                       mode = 'val', random_split=args.random_split)
            #print ("val performance", val_performance, "best val performance : ", best_val_performance)

            if val_performance > best_val_performance:
                # print ("saving best model ... at ", val_performance)
                best_val_performance = val_performance
                torch.save(model.state_dict(), checkpoint_path)

        print ("Epoch {}: Mean Loss: {}".format(epoch, epoch_loss))
        print ()

    model = models.BPR(**model_params)
    model.load_state_dict(torch.load(checkpoint_path))
    if args.use_gpu:
        model = model.to(device)

    model.eval()
    print ("Evaluating on test set: ")
    evaluate(test_set, train_set, model, 
             use_gpu=args.use_gpu, mode = 'test', random_split=args.random_split)
