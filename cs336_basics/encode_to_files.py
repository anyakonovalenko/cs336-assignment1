from tokenizer import Tokenizer
import numpy as np

vocab_path = '/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_vocab_10k.pkl'
merges_path = '/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_merges_10k.pkl'
data_dir = '/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/data/'


def encode_to_files(input_path, output_path, vocab_path, merges_path):
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merges_path,
        ['<|endoftext|>', '<|end|>']
    )

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
        tokens = tokenizer.encode(text)
        np.save(output_path, np.array(tokens, dtype=np.int32))

encode_to_files(
    f'{data_dir}TinyStoriesV2-GPT4-train.txt',
    'tokens/tokens_train',
    vocab_path,
    merges_path
)

# encode_to_files(
#     f'{data_dir}TinyStoriesV2-GPT4-valid.txt',
#     'tokens/tokens_valid',
#     vocab_path,
#     merges_path
# )