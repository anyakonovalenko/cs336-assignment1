import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from timeit import default_timer as timer
from collections import Counter
import pickle
import cProfile
import pstats

# NOTES
# -----------------------
# 'cat'.encode('utf-8') == bytes([char for char in chunk])
# ([bytes([char]) for char in 'cat'.encode('utf-8') ]) == sequence of elements in bytes

# for token in sorted_special_tokens:
#self.special_tokens_pattern += "(" + re.escape(token) + ")|"
# ----------------------------
# NOTES

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.len_special = len(special_tokens) if special_tokens else None
        self.reverse_vocab = {value:key for key, value in vocab.items()}
        self.extended_merges = {pair: self.reverse_vocab[pair[0]+pair[1]] for pair in merges}

        #special tokens for encoding
        #1. put special tokens into vocab
        #2. Create string as a regex string to be able to split by them (re.escape, re.split)
        #3. Include () in the string to have a special token incleded into partition
        #4. find them in text and split by them and not delete them(separate token)

        if self.special_tokens:
            idx = len(vocab)
            for i, token in enumerate(self.special_tokens):
                if token not in self.reverse_vocab:
                    self.reverse_vocab[idx+i] = token.encode('utf-8')
                    self.vocab[token.encode('utf-8')] = idx+i
        if self.special_tokens:
            escaped_tokens = [re.escape(token)for token in sorted(special_tokens, key = len, reverse=True)]
            self.special_tokens_pattern = '(' + '|'.join(escaped_tokens) + ')'
        else:
            self.special_tokens_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens = None):
        # cls refers to BPETokenizer (the class itself)
        try:
            with open(vocab_filepath, 'rb') as f:
                vocab = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_filepath}")

        try:
            with open(merges_filepath, 'rb') as f:
                merges = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Merges file not found: {merges_filepath}")

        return cls(vocab, merges, special_tokens)

    def get_pairs(self, word):
        pairs = Counter()
        for i in range(len(word) - 1):
            key_two = (word[i], word[i + 1])
            pairs[key_two] += 1
        return pairs

    def encode(self, text):
        PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # tokens = list(text.encode("utf-8"))
        new_parts = []
        if self.special_tokens_pattern:
            parts = re.split(self.special_tokens_pattern, text)
        else:
            parts = [text]

        for part in parts:

            if not part:
                continue

            if self.special_tokens and part in self.special_tokens:
                new_parts.append([part.encode("utf-8")])
                continue

            tokens = []
            for match in re.finditer(PAT, part):
                chunk = match.group()
                chunk_ids = [bytes([char]) for char in chunk.encode("utf-8")]
                tokens.append(chunk_ids)

            for num, token in enumerate(tokens):
                while True:
                    pairs = self.get_pairs(token)

                    if not pairs:
                        break

                    # find the best pair
                    best_pair = None
                    best_pair_idx = float('inf')

                    for pair in pairs.keys():
                        if pair in self.merges:
                            merge_idx = self.extended_merges[pair]
                            if merge_idx < best_pair_idx:
                                best_pair_idx = merge_idx
                                best_pair = pair
                    if best_pair == None:
                        break
                    # merge the best pair
                    new_token = []
                    i = 0
                    while i < len(token):
                        if i < len(token) - 1 and (token[i], token[i+1]) == best_pair:
                            new_token.append(token[i] + token[i+1])
                            i += 2
                        else:
                            new_token.append(token[i])
                            i += 1
                    token = new_token
                new_parts.append(token)
                # if len(token) <= 1:
                #     tokens[num] = token
                # else:
                #     tokens[num] = token

        result = []
        for pre_token in new_parts:  # Each pre_token is like [b'the']
            for byte_sequence in pre_token:  # Each byte_sequence is like b'the'
                result.append(self.reverse_vocab[byte_sequence])
        return result

    def decode(self, ids):
        text_bytes = b"".join([self.vocab[i] for i in ids])
        return text_bytes.decode("utf-8",
                                 errors='replace')  # for example 128 is a continuation byte, not the first one, so we need to do it as it can come from LLM and we will not be able to decode
    def encode_iterable(self, iterable):
        for text in iterable:
            # Encode each string and yield its tokens one by one
            token_ids = self.encode(text)
            for token_id in token_ids:
                yield token_id



if __name__ == "__main__":
    tokenizer = Tokenizer.from_files('/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_vocab_10k.pkl','/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/tinystories_merges_10k.pkl', ['<|endoftext|>', '<|end|>'])
    a = tokenizer.encode('"Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"')
    print(tokenizer.decode(a))
