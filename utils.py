import pandas as pd
import scipy.sparse as sp
import numpy as np
import networkx as nx
from torch.utils import data
import torch
from sklearn.metrics.pairwise import cosine_similarity


def load_social():
    f_path = 'data/social.csv'
    raw_social = {}
    with open(f_path) as f:
        next(f)
        for line in f:
            u1, u2 = line.strip().split(',')
            u1, u2 = int(u1), int(u2)
            try:
                raw_social[u1].append(u2)
            except KeyError:
                raw_social[u1] = [u2]
        f.close()
        
    social = {key: torch.tensor(val) for key, val in raw_social.items()}
    
    return social

def load_category():
    f_path = 'data/item_cats.csv'
    cats = []
    raw_cats = {}
    with open(f_path) as f:
        next(f)
        for line in f:
            item, cat = line.strip().split(',')
            raw_cats[int(item)] = cat
            cats.append(cat)
            
        f.close()
    cats = list(set(cats))
    social = {key: cats.index(val) for key, val in raw_cats.items()}
    
    return social

def load_negative(num_users=6858, num_items=3317):
    f_path = 'data/train.csv'
    total = set(np.arange(num_items))
    user_neg = {x: set() for x in range(num_users)}
    with open(f_path) as f:
        next(f)
        for line in f:
            user, item = line.strip().split(',')
            user_neg[int(user)].add(int(item))
            
        f.close()
    
    user_neg = {key: torch.tensor(list(total.difference(val))) for key, val in user_neg.items()}
    
    return user_neg

def load_embedding(num_users=6858, num_items=3317, 
                   embed_dim=512, random_split = True):
    if random_split:
        f_path = 'data/train_emb_{}.txt'.format(embed_dim)
    else:    
        f_path = 'data/all_emb_{}.txt'.format(embed_dim)
    user_emb = np.zeros((num_users, embed_dim))
    item_emb = np.zeros((num_items, embed_dim))
    with open(f_path) as f:
        next(f)
        for line in f:
            l = line.strip().split()
            if '10000' in l[0]:
                item_emb[int(l[0][5:]), :] = np.array([float(x) for x in l[1:]])
            else:
                user_emb[int(l[0]), :] = np.array([float(x) for x in l[1:]])
            
        f.close()
        
    social_path = 'data/social_emb_{}.txt'.format(embed_dim)
    social_emb = np.zeros((num_users, embed_dim))
    with open(social_path) as f:
        next(f)
        for line in f:
            l = line.strip().split()
            social_emb[int(l[0]), :] = np.array([float(x) for x in l[1:]])
            
        f.close()
    
    social_sim = cosine_similarity(social_emb)/5082
    np.fill_diagonal(social_sim, 0)
    social_sim = torch.FloatTensor(social_sim)
        
    return user_emb, item_emb, social_sim
                

def load_data(dataset_str, random_split=True):
    data_dir = 'data/'
    df_train = pd.read_csv(data_dir + "train.csv")
    # df_test = pd.read_csv(data_dir + "test.csv")
    num_items = max(df_train.item.unique()) + 1
    num_users = max(df_train.user.unique()) + 1
    df_test = df_train.sample(frac=0.2)

    if random_split:
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_interactions = df_train.drop(df_test.index)[['user', 'item']].values
        train_set = df_train.drop(df_test.index)[['user', 'item']].groupby(['user'])['item'].apply(list).reset_index()
    else:
        train_interactions = df_train[['user', 'item']].values
        test_set = df_test.groupby(['user'])['item'].apply(list).reset_index()
        train_set = df_train[['user', 'item']].groupby(['user'])['item'].apply(list).reset_index()
    
    df_test.to_csv(data_dir+'test.csv',index=False)
    return train_interactions, train_set, test_set, int(num_items), int(num_users)


class Dataset(data.Dataset):
    def __init__(self, ratings):
        self.ratings = ratings

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        user, item = self.ratings[idx]
        return user, item

if __name__ == "__main__": 
  load_data('yelp')