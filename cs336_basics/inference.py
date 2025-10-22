import torch

from tokenizer import Tokenizer
from checkpoint import load_checkpoint
from transformer_lm import TransformerLM
from AdamW import AdamW
from einops import rearrange, einsum
import random


vocab_path = '/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_vocab_10k.pkl'
merges_path = '/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_merges_10k.pkl'



def decoding(x, max_count_generated_tokens = 20, temperature = 0.8, max_context_length = 256, p = 0.95):
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        ['<|endoftext|>', '<|end|>']
    )
    tokens = tokenizer.encode(x)
    model = TransformerLM(vocab_size = 10000, context_length = max_context_length, d_model = 512, num_layers = 4, num_heads = 16, d_ff = 1344, theta = 10000, device=None)
    optimizer = AdamW(model.parameters())
    _ = load_checkpoint('tokens/model', model, optimizer)
    model.eval()
    #length of input changes dynamically up to context length
    if len(tokens) > max_context_length:
        tokens = tokens[-max_context_length]
    tokens = rearrange(torch.tensor(tokens), 'seq -> 1 seq')

    for i in range(max_count_generated_tokens):

        with torch.no_grad():
            pred = model(tokens)
            pred = rearrange(pred, '1 seq vocab-> seq vocab')
            pred = pred[-1]
            logits = torch.exp(pred/temperature)/torch.sum(torch.exp(pred/temperature))
            max_idx = []
            sum_prob = 0
            while sum_prob <= p:
                prob, idx = torch.max(logits, dim=0)
                sum_prob += prob
                max_idx.append(idx)
                idx = random.choice(max_idx)
            idx = rearrange(idx, '-> 1 1')
            tokens = torch.cat([tokens, idx], dim=1)
            generated_token = tokenizer.vocab[int(idx)].decode("utf-8")
            print(generated_token)



decoding("He is very")