from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import math
import os
from tqdm import tqdm
from transformers import GPT2Tokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
#import multiprocessing as mp
import torch.multiprocessing as mp

# FIX LOOP BREAKING (incorrect loss division)

MINI_BATCH_SIZE = 4 #16
BATCH_SIZE = 2**15 # 2**19
TOKEN_LENGTH = 128 #1024
USE_DDP = True
WORLD_SIZE = 8

# ----------
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
@dataclass
class GPTConfig:
    seq_length: int = TOKEN_LENGTH
    vocab_size: int = 50257
    num_heads: int = 12
    embedding_dim: int = 768
    num_blocks: int = 12
    batch_size: int = MINI_BATCH_SIZE

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
        #self.register_buffer('mask', torch.tril(torch.ones(sl, sl)).view(1, 1, sl, sl))

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

        #attn = q @ k.transpose(2, 3) / math.sqrt(k.size(-1)) # batch, head dim, T, T
        #attn = attn.masked_fill(self.mask[:,:,:t,:t]==0, float('-inf')) # trim mask to current sequence length
        #attn = F.softmax(attn, dim=-1)
        #out = attn @ v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
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
        self.eval()
        tokens = input_tokens

        if seq_length is None:
            seq_length = self.config.seq_length
        while tokens.size(1) < seq_length:
            with torch.no_grad():
                  logits = self(tokens)[:, -1, :] # get last time dimension
                  probs = F.softmax(logits, dim=-1)
                  topk_probs, topk_idxs = torch.topk(probs, 20, dim=-1)
                  idx_ix = torch.multinomial(topk_idxs.to(torch.float32), 1) #
                  idx = topk_idxs.index_select(dim=-1, index=idx_ix.flatten()) #
                  tokens = torch.cat((tokens, torch.tensor([idx], device=tokens.device).to(tokens.dtype).view(1, 1)), dim=-1) # still b, t
        self.train() # assuming we continue training afterwards
        return tokens

class DataLoader:
    def __init__(self, b, t):
        self.batch_size = b
        self.t = t

        self.corpus = []
        #shards = [f'datashards/shard_{i}.npy' for i in range(100)]
        shards = [f'datashards/shard_{1}.npy']
        for shard in shards:
            self.corpus.extend(np.load(shard).tolist())
        
        print(f'Loaded corpus of length {len(self.corpus)}')
        self.tokens = torch.tensor(self.corpus, dtype=torch.long)

        self.current_batch = 0

    def next_batch(self):
        tokens = self.tokens[self.current_batch:self.current_batch + self.batch_size*self.t + 1]
        inputs = tokens[:-1].view(self.batch_size, self.t)
        labels = tokens[1:].view(self.batch_size, self.t)

        self.current_batch += self.batch_size*self.t
        if self.current_batch + self.batch_size*self.t + 1 > len(self.corpus):
            return None, None

        return inputs, labels

def test_run(gpt, device):
    model = gpt.module # if gpt is a DDP
    input = "I am a language model"
    input = tokenizer(input).input_ids
    tokens = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    out = model.generate(tokens, seq_length=32).detach().cpu()
    out = tokenizer.decode(out.flatten().tolist(), skip_special_tokens=True)
    print(out)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    try:
        setup(rank, world_size)
        device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(device)
        print(f'Setting up device {device}')
    
        config = GPTConfig()
        gpt = GPT(config, device).to(device)
        gpt = torch.compile(gpt)
        gpt = DDP(gpt, device_ids=[rank])
    
        dataloader = DataLoader(MINI_BATCH_SIZE, TOKEN_LENGTH)
    
        batch_size = BATCH_SIZE # close to .5M as in GPT3
        assert batch_size % MINI_BATCH_SIZE == 0, 'batch size must be a divisor of 2**19'
        grad_steps = int(batch_size // MINI_BATCH_SIZE)
    
        # optimizer decay
        def get_lr(iter, warmup=10, max_steps=50, max_lr=1e-3, min_lr=1e-4):
            if iter < warmup:
                return max_lr * (iter+1) / warmup # linear schedule
            elif iter > warmup:
                return min_lr
            ratio = (iter - warmup) / (max_steps - warmup)
            c = math.cos(ratio * math.pi/2)
            return min_lr + c * (max_lr - min_lr)
    
        optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0005, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda iter: get_lr(iter))
    
        num_epochs = 100
    
        for epoch in range(num_epochs):
            test_run(gpt, device)
            torch.cuda.empty_cache()
            epoch_loss = 0
            print(f'Rank {rank}: Epoch {epoch+1} of {num_epochs}')
            data_is_loading = True
            num_epoch_batches = 0
            dataloader.current_batch = 0
            
            while data_is_loading:
                print(f'Machine {rank} initialising batch')
                batch_loss = 0
                num_epoch_batches += 1
                for step in tqdm(range(grad_steps)):
                    inputs, labels = dataloader.next_batch()
                    if inputs == None:
                        data_is_loading = False
                        break # works if len(dataloader.corpus) >> batch_size, eg. at least ~1B tokens
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
    
                    with autocast(device_type='cuda', dtype=torch.float16): #bfloat16
                        logits = gpt(inputs)
                        labels = F.one_hot(labels, num_classes=config.vocab_size).float()
                        loss = F.cross_entropy(logits, labels) / grad_steps # adjust loss scaling
                    loss.backward()
                    batch_loss += loss.item()
    
                # this will execute after the break
                torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
                optimizer.step()
                torch.cuda.synchronize()
                epoch_loss += batch_loss
                print(f'Batch loss: {batch_loss}')
    
            scheduler.step()
            if epoch % 10 == 0 and rank == 0:
                torch.save(gpt.state_dict(), f'weights/gpt_weights_{epoch}.pth')
            print(f'Rank {rank}: Loss: {(epoch_loss/num_epoch_batches):.4f}, Learning rate {scheduler.get_last_lr()[0]:.4f}')
    except Exception as e:
        print(f"[Rank {rank}] Error during training: {e}")
    cleanup()
    
if __name__ == '__main__':
    if USE_DDP:
        mp.spawn(train, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    else:
        print("Use model.py to train without DDP")
        quit()
