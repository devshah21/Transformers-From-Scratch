import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    # refer to Notion for annotated code
    def __init__(self, embed_size, heads):
        # in the paper, the embedddings are split into parts, the number of parts is determined by heads; line 11
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (self.head_dim * heads == embed_size) # sanity check to ensure values allign
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False) # map the head dimensions to the head dimensions
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False) # same thing again here
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False) 
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size) # fully connected out is mapping embed size to embed size
    
    def forward(self, values, keys, query, mask):
        N = query.shape[0] # number of training examples we send in at the same time
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1] # source and target sentence length
        
        # split the embeddings into self.head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim) 
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)
        
        eng = torch.einsum("nqhd,nkhd->nhqk")
        # eng shape: (N, heads, query_len, key_len)
        