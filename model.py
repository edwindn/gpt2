from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from transformers import GPT2Tokenizer

@dataclass
class GPTConfig:
    seq_length: int = 1024
    vocab_size: int = 50257
    num_heads: int = 12
    embedding_dim: int = 768
    num_blocks: int = 12
    batch_size: int = 128

def sinusoidal_encoding(seq_len, dim, max_timescale=10000):
    PE = np.empty((dim, seq_len))
    pos = np.arange(seq_len).reshape(1, -1)
    i = np.arange(dim).reshape(-1, 1)
    inv = max_timescale ** (2/dim * i//2)
    PE[::2,:] = np.sin(pos / inv[::2])
    PE[1::2,:] = np.cos(pos / inv[1::2])
    return torch.tensor(PE)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(config.embedding_dim, 4*config.embedding_dim), # blows up dim
            nn.GELU(),
            nn.Linear(4*config.embedding_dim, config.embedding_dim)
        )

    def forward(self, x):
        return self.main(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.qkv_proj = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        sl = config.seq_length
        self.register_buffer('mask', torch.tril(torch.ones(sl, sl)).view(1, 1, sl, sl))

        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim

        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        b, t, c = x.size() # batch, seq length, size

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        q = q.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, head dim, seq length (T), size
        k = k.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        attn = q @ k.transpose(2, 3) / math.sqrt(k.size(-1)) # batch, head dim, T, T
        attn = attn.masked_fill(self.mask[:,:,:t,:t]==0, float('-inf')) # trim mask to current sequence length
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(b, t, c)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attention(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x

class GPT(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)
        self.pos_emb = sinusoidal_encoding(config.seq_length, config.embedding_dim).T.to(device)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_blocks)).to(device)
        self.ln = nn.LayerNorm(config.embedding_dim).to(device)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False).to(device)
        
        self.token_emb.weight = self.lm_head.weight # input and output embeddings use the same transformation

    def forward(self, tokens):
        b, t = tokens.size()
        assert t <= self.config.seq_length, "Cannot convert tokens longer than max sequence length"
        x = self.token_emb(tokens) # b, t -> b, t, c
        pos_emb = self.pos_emb[:t,:]
        x = (x + pos_emb).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        return self.lm_head(x) # logits

    def generate(self, input_tokens, seq_length=None):
        tokens = input_tokens

        if seq_length is None:
            seq_length = self.config.seq_length
        while tokens.size(1) < seq_length:
            with torch.no_grad():
                  logits = gpt(tokens)[:, -1, :] # get last time dimension
                  probs = F.softmax(logits, dim=-1)
                  topk_probs, topk_idxs = torch.topk(probs, 20, dim=-1)
                  idx_ix = torch.multinomial(topk_idxs.to(torch.float32), 1) #
                  idx = topk_idxs.index_select(dim=-1, index=idx_ix.flatten()) #
                  tokens = torch.cat((tokens, torch.tensor([idx], device=tokens.device).to(tokens.dtype).view(1, 1)), dim=-1) # still b, t
        
        return tokens

class DataLoader:
    def __init__(self, b, t):
        self.batch_size = b
        self.t = t
        corpus = open('input.txt', 'r').read().strip().replace('\n\n', '\n')
        corpus = tokenizer(corpus).input_ids
        self.tokens = torch.tensor(corpus, dtype=torch.long)

        self.current_batch = 0

    def next_batch(self):
        tokens = self.tokens[self.current_batch:self.current_batch + self.batch_size*self.t + 1]
        inputs = tokens[:-1].view(self.batch_size, self.t)
        labels = tokens[1:].view(self.batch_size, self.t)

        self.current_batch += self.batch_size*self.t
        if self.current_batch + 1 > len(corpus):
            self.current_batch = 0

        return inputs, labels

def test_run():
    input = "I am a language model"
    input = tokenizer(input).input_ids
    tokens = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    out = gpt.generate(tokens, seq_length=32).detach().cpu()
    out = tokenizer.decode(out.flatten().tolist(), skip_special_tokens=True)
    print(out)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    config = GPTConfig()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #dataloader = DataLoader(config.batch_size, config.seq_length)
    gpt = GPT(config, device).to(device)
    dataloader = DataLoader(64, 32)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0005)

    test_run()

    num_epochs = 100
    batches_per_epoch = 1615 # about 10 full runs

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} of {num_epochs}')
        for _ in tqdm(range(batches_per_epoch)):
            torch.cuda.empty_cache()
            inputs, labels = dataloader.next_batch()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = gpt(inputs)
            labels = F.one_hot(labels, num_classes=config.vocab_size).float()
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        torch.save(gpt.state_dict(), f'weights/gpt_weights_{epoch}.pth')

    test_run()



    
