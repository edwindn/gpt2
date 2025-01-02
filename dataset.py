from datasets import load_dataset, Dataset
from transformers import GPT2Tokenizer
import numpy as np
import multiprocessing as mp

# !! mkdir ../datashards && mkdir weights

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT")['train']

def tokenize(example):
    tokens = tokenizer(example['text']).input_ids
    return {'input_ids': tokens}

# each shard has 100M tokens
length = ds.num_rows #9672101
shard_length = length // 100

eot = tokenizer("<|endoftext|>").input_ids[0]

def save_shard(idx):
    shard = Dataset.from_dict(ds[idx*shard_length:(idx+1)*shard_length])
    shard = shard.map(tokenize, batched=True)
    tokens = [eot]
    for arr in shard['input_ids']:
        tokens.extend(arr)
    tokens.append(eot)
    tokens = np.array(tokens).astype(np.uint16)
    np.save(f'../datashards/shard_{idx}.npy', tokens)
    print(f'Saved shard at ../datashards/shard_{idx}.npy')

if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(save_shard, range(100))


