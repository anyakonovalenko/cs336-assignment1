import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from timeit import default_timer as timer
from collections import Counter
import pickle
import cProfile
import pstats

#20 minutes

def get_stats(word_frequency):
    counts = Counter()
    for word, frequency in word_frequency.items():
        for i in range(len(word) - 1):
            key_two = (word[i], word[i + 1])
            counts[key_two] += frequency
    return counts

def merge_bpe(ids, pair, idx):
    newids = [] # update text in token with a new pair
    for chunk_id in range(len(ids)):
        newids_chunk = []
        i = 0
        while i < len(ids[chunk_id]) - 1:
            if ids[chunk_id][i] == pair[0] and ids[chunk_id][i + 1] == pair[1]:
                newids_chunk.append(idx)
                i += 2
            else:
                newids_chunk.append(ids[chunk_id][i])
                i += 1
        if i == len(ids[chunk_id]) - 1:
            newids_chunk.append(ids[chunk_id][i])
        newids.append(newids_chunk)

    return newids


def process_chunk(args):
    start, end, filepath, PAT, num_special, special_tokens = args

    # Read this chunk
    with open(filepath, 'rb') as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")

    # Apply your existing regex logic

    delimiter = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(delimiter, chunk_text)
    word_frequency = Counter()
    for doc in documents:
        if doc.strip():

            for match in re.finditer(PAT, doc):
                chunk = match.group()
                chunk_ids = tuple([byte_val + num_special for byte_val in chunk.encode("utf-8")])
                word_frequency[chunk_ids] += 1
    return word_frequency


def trainingBPE(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = timer()
    PAT = re.compile( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    num_special = len(special_tokens)
    # for match in re.finditer(pattern, text):
    # text = "The quick é brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once."

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    print("opened file ", timer() - start_time)
    chunk_args = [(start, end, input_path, PAT, num_special, special_tokens)
                  for start, end in zip(boundaries[:-1], boundaries[1:])]
    print("split chunks ", timer() - start_time)
    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    print("pool map done ", timer() - start_time)
    # Flatten results into single ids list
    from collections import defaultdict
    word_frequency = defaultdict(int)
    for chunk_ids in chunk_results:
        for k, v in chunk_ids.items():
            word_frequency[k] += v
    print("ids extend ", timer() - start_time)

    num_merges = vocab_size - 256 - len(special_tokens)
    merges = {} #(int, int) -> int  you had othervise previously newid -> (pair (int, int))
    merges_list = []
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')

    for i in range(256):
        vocab[num_special + i] = bytes([i])

    stats = get_stats(word_frequency)

    print("got stats ", timer() - start_time)
    for merge_id in range(num_merges):
        # new_ids = []
        # Use vocab to get byte representation for lexicographic comparison
        pair = max(stats, key=lambda p: (stats[p], (vocab[p[0]]), vocab[p[1]]))
        iter_num = num_special + 256 + merge_id
        vocab[iter_num] = vocab[pair[0]] + vocab[pair[1]]  # Add new token to vocab
        new_word_frequency = {}
        for word, freq in word_frequency.items():
            new_word = []
            #update stats
            i = 0
            length = len(word)
            while i < length:
                if i < length - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    if i > 0:
                        stats[(word[i-1], pair[0])] = stats.get((word[i-1], pair[0]), 0) - freq
                        stats[(word[i-1], iter_num)] = stats.get((word[i-1], iter_num), 0) + freq

                    if i + 2 < length:
                        stats[(pair[1], word[i + 2])] = stats.get((pair[1], word[i + 2]), 0) - freq
                        stats[(iter_num, word[i+2])] = stats.get((iter_num, word[i+2]), 0) + freq
                    stats[(pair[0], pair[1])] = stats.get((pair[0], pair[1]), 0) - freq
                    if stats[(pair[0], pair[1])] == 0:
                        del stats[(pair[0], pair[1])]
                    new_word.append(iter_num)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_frequency[tuple(new_word)] = freq
        word_frequency = new_word_frequency
        merges[pair] = iter_num

    for (m1, m2), value in merges.items():
        merges_list.append((vocab[m1], vocab[m2]))

    print("full run ", timer() - start_time)
    return vocab, merges_list

def analyze_vocab(vocab):
    longest_token = sorted(vocab.values(), key=len, reverse=True)[1]
    return longest_token
    # return max(vocab.values(), key=lambda x: len(x))



def main():
    #train to tiny stories
    input_path = "/Users/anko/Documents/Study/cs336-assignment1/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    start_time = timer()
    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = trainingBPE(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )
    total_time = timer() - start_time
    print('total_time', total_time)
    print('longest element', analyze_vocab(vocab))
    profiler.disable()

    vocab_path = "tinystories_vocab_10k.pkl"
    merges_path = "tinystories_merges_10k.pkl"

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    with open(merges_path, 'wb') as f:
        pickle.dump(merges, f)

    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Show top 20 functions

if __name__ == "__main__":
    main()