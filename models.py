import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import *

class BPR(nn.Module):
    def __init__(self, num_items, num_users, embed_dim, social_dict, user_neg, 
                 user_emb, item_emb, social_sim, social, lambda_u=0.01, 
                 lambda_i=0.01, method='BPR', use_gpu=0, hidden_size = 250):
        super(BPR, self).__init__()
        self.num_items = num_items
        self.num_users = num_users
        print ("# users", self.num_users, "# items", self.num_items)
        self.embed_dim = embed_dim
        self.method = method
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.use_gpu = use_gpu
        self.social = social
        self.user_neg = user_neg
        self.social_dict = social_dict
        
        self.user_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(user_emb))
        self.user_embeddings.weight.requires_grad=True
        self.item_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(item_emb))
        self.item_embeddings.weight.requires_grad=True
        self.social_weight = nn.Parameter(torch.FloatTensor(social_sim),
                                          requires_grad=True)
        
    
    def forward(self, users, items):
        assert users.shape[0] == items.shape[0]
        batch_size = users.shape[0]

        if self.method == 'BPR':
            batch_user_embeddings = self.user_embeddings(users)  # [B, D]
        # user_embeddings = [B, D]
        batch_item_embeddings = self.item_embeddings(items)  # [B, D]
        self.batch_user_embeddings = batch_user_embeddings
        # [B, D] x [B, D, 1] -> [B, 1] -> [B]
        positive_predictions = torch.bmm(batch_user_embeddings.view(batch_size, 1, self.embed_dim),
                                         batch_item_embeddings.view(batch_size, self.embed_dim,
                                                                    1)).squeeze()

        if self.social:
            batch_social_weights = self.social_weight[users]
                
            batch_social_embeddings = torch.cat([self.user_embeddings.weight]*batch_size, 0)
            batch_social_scores = torch.bmm(batch_social_embeddings.view(batch_size, 
                                                                         self.num_users, 
                                                                         self.embed_dim), 
                                batch_item_embeddings.view(batch_size, self.embed_dim,1))
            social_influence = torch.bmm(batch_social_weights.view(batch_size, 1, self.num_users), 
                                         batch_social_scores).squeeze()
        
            positive_predictions += social_influence
        
        return positive_predictions

    def sample_negs(self, items, num_negs):
        #batch_size = items.shape[0]
        #negative_items = torch.randint(0, self.num_items, (batch_size, num_negs))  # [B, 10]
        negative_items = torch.tensor(list(map(lambda x: self.user_neg[int(x)][torch.randint(0, len(self.user_neg[int(x)]), (1,))][0], items)))
        return negative_items


    def loss(self, positive_predictions , negative_predictions):
        # BPR loss.
        bpr_loss =  - torch.log(torch.sigmoid(positive_predictions -
                                negative_predictions)).mean()
                
        # regularization loss.
        if self.method == 'BPR':
            bpr_loss += self.lambda_u * torch.norm(self.user_embeddings.weight)**2
            bpr_loss += self.lambda_i * torch.norm(self.item_embeddings.weight)**2

        return bpr_loss
