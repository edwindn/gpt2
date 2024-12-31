import torch
from tqdm import tqdm
from dataclasses import dataclass
from transformers import GPT2Tokenizer
from model import GPTConfig
import torch.nn.functional as F
print("Ensure hugging face authentication before running")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

repo = "edncodeismad/gpt2_dataset"

corpus = open('input.txt', 'r').read().strip().replace('\n\n', '\n')
corpus = tokenizer(corpus).input_ids

maxlen = 323 # default: 323

corpus = corpus[:maxlen*1024 + 1]
corpus = torch.tensor(corpus, dtype=torch.long)
x = corpus[:-1].view(maxlen, 1024)
labels = corpus[1:].view(maxlen, 1024)

config = GPTConfig()

x_seq = []
label_seq = []

with torch.no_grad():
    for i in tqdm(reversed(range(1024))):
        x_seq.append(x[:, :i].cpu().numpy())
        label_seq.append(F.one_hot(labels[:, :i], num_classes=config.vocab_size).float().cpu().numpy())

dataset = Dataset.from_dict({"x": x_seq, "label": label_seq})
dataset.push_to_hub(repo)
