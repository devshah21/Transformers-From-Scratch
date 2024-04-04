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
        
        # nn.Linear basically does weight * input + bias --> y = mx +b 
        
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
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        eng = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) 
        # eng shape: (N, heads, query_len, key_len)
        # queries shape: N, query_len, heads, heads_dim
        # keys shape: N, keys_len, heads, head_dim
        # energy shape: N, queries_len, key_len
        if mask is not None:
            eng = eng.masked_fill(mask == 0, float("-1e20")) # mask for target is a traingular matrix; when closing, we replace element with a float
        
        attention = torch.softmax(eng / (self.embed_szie ** (1/2)), dim=3) # dim = 3 is used to normalize across the source sentence
        
        out = torch.einsum("nhql,nlhd->", [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: N, heads, query_len, key_len
        # values shape: N, values_len, heads, heads_dim
        # after einsum (N, query_len, heads, heads_dim) then flatten last two dimensions
        
        out = self.fc_out(out)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_exp):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size) # layer norm takes the average of every single example and normalizes 
        self.norm2 = nn.LayerNorm(embed_size)
        
        # feedforward part:
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_exp * embed_size),
            nn.ReLU(),
            nn.Linear(forward_exp * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cpu",
        max_length=100,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )

        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out
        
            