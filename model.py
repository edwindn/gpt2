from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

@dataclass
class GPTConfig:
    seq_length: int = 1024
    vocab_size: int = 50257
    num_heads: int = 12
    embedding_dim: int = 768
    num_blocks: int = 12
