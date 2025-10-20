import torch
import torch.nn as nn
from einops import rearrange, einsum
from cs336_basics.softmax import softmax
import math
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np
import argparse


import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
from data_loading import data_loading
from transformer_block import TransformerBlock
from cross_entropy import cross_entropy
from AdamW import AdamW
from get_lr_cosine_schedule import get_lr_cosine_schedule
from gradient_clipping import gradient_clipping
from checkpoint import load_checkpoint, save_checkpoint


# W.grad tells you how much the loss changes when you change W. (slope)


def train(training_path, validation_path, batch_size, context_length, vocab_size,
          d_model, num_layers, num_heads, d_ff, theta, max_learning_rate, min_learning_rate,
          warmup_iters, cosine_cycle_iters, max_l2_norm, max_steps, path_checkpoint_model, resume, device):
    dataset_train = np.load(training_path, mmap_mode='r')
    dataset_val = np.load(validation_path, mmap_mode='r')
    transformer = TransformerBlock(vocab_size, context_length, d_model, num_layers, num_heads, d_ff, theta, device)
    optimizer = AdamW(transformer.parameters())
    starting_step = 0

    if resume:
        starting_step = load_checkpoint(path_checkpoint_model, transformer, optimizer)

    for i in range(starting_step, max_steps):
        inputs, targets = data_loading(dataset_train, batch_size, context_length, device)  # batch_size, context_length
        lr = get_lr_cosine_schedule(i, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad()  #The gradient keeps accumulating like a snowball in backward, Each iteration, you're at a different position (different W values), so you calculate a different slope (different gradient)
        values = transformer.forward(inputs)
        loss = cross_entropy(values, targets)
        if (i % 50 == 0):
            print(f'Training Loss iter {i}', loss.item())
        loss.backward()
        gradient_clipping(params=transformer.parameters(), max_l2_norm=max_l2_norm)
        optimizer.step()
        if (i % 100 == 0):
            save_checkpoint(transformer, optimizer, i, path_checkpoint_model)
            with torch.no_grad():
                inputs, targets = data_loading(dataset_val, batch_size, context_length, device)  # batch_size, context_length
                values = transformer.forward(inputs)
                loss = cross_entropy(values, targets)
                print(f'Validation Loss iter {i}', loss.item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a transformer language model')

    # Data parameters
    parser.add_argument('--training_path', type=str, required=True, help='Path to training data')
    parser.add_argument('--validation_path', type=str, required=True, help='Path to validation data')

    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=50257, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=256, help='Context length')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feed-forward dimension')
    parser.add_argument('--theta', type=int, default=10000, help='Theta for Rope')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_learning_rate', type=float, default=6e-4, help='Maximum learning rate')
    parser.add_argument('--min_learning_rate', type=float, default=6e-5, help='Minimum learning rate')
    parser.add_argument('--warmup_iters', type=int, default=100, help='Number of warmup iterations')
    parser.add_argument('--cosine_cycle_iters', type=int, default=1000, help='Cosine cycle iterations')
    parser.add_argument('--max_l2_norm', type=float, default=1.0, help='Max L2 norm for gradient clipping')
    parser.add_argument('--max_steps', type=int, default=10000, help='Maximum training steps')

    #Checkpoint parameters
    parser.add_argument('--path_checkpoint_model', type=str, required=True, help='Path to saving/loading the model')
    parser.add_argument('--resume', action='store_true', help='Resume training or not, special handling for flags, True with --resume')

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')

    args = parser.parse_args()

    train(
        training_path=args.training_path,
        validation_path=args.validation_path,
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        max_l2_norm=args.max_l2_norm,
        max_steps=args.max_steps,
        path_checkpoint_model=args.path_checkpoint_model,
        resume=args.resume,
        device=args.device
    )



