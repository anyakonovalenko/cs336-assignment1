import regex as re
from multiprocessing import Pool
from cs336_basics.pretokenization_example import find_chunk_boundaries
from timeit import default_timer as timer
from collections import Counter

def get_stats(ids, counts=None):
    counts = {} if counts is None else counts
    for i in range(len(ids) - 1):
        key_two = (ids[i], ids[i + 1])
        if key_two in counts:
            counts[key_two] += 1
        else:
            counts[key_two] = 1
    return counts

def merge_bpe(ids, pair, idx):
    start_time = timer()
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

# def decode_my(ids, merges):
#     print(merges.values())
#     for i, merge in sorted(merges.items(), key = lambda x: x[1], reverse = True): # later merges depend on earlier tokens
#         new_ids = []
#         for id in ids:
#             if id == merge:
#                 new_ids.append(i[0])
#                 new_ids.append(i[1])
#             else:
#                 new_ids.append(id)
#         ids = new_ids
#     text = bytes(ids).decode('utf-8')  # This gives you 'é'
#     return text


def decode_karp(ids, merges):
    vocab = {i: bytes([i]) for i in range(256)}
    for (m1, m2), value in merges.items(): # it will be added in order of creating
        vocab[value] = vocab[m1] + vocab[m2] #so it will always be accassible in vocabulary, for later ids iw t will be longer
    text_bytes = b"".join([vocab[i] for i in ids])
    return text_bytes.decode("utf-8", errors= 'replace')  # for example 128 is a continuation byte, not the first one, so we need to do it as it can come from LLM and we will not be able to decode

def encode(text, merges):
    tokens = list(text.encode("utf-8"))
    for (m1, m2), value in merges.items():
        new_tokens = []
        j = 0
        while j < len(tokens):
            if j < len(tokens)-1 and tokens[j] == m1 and tokens[j+1] == m2:
                new_tokens.append(value)
                j += 2
            else:
                new_tokens.append(tokens[j])
                j += 1
        tokens = new_tokens
    return tokens


def process_chunk(args):
    start, end, filepath, PAT, num_special, special_tokens = args

    # Read this chunk
    with open(filepath, 'rb') as f:
        f.seek(start)
        chunk_text = f.read(end - start).decode("utf-8", errors="ignore")

    # Apply your existing regex logic

    delimiter = "|".join(re.escape(token) for token in special_tokens)
    documents = re.split(delimiter, chunk_text)

    all_ids = []
    for doc in documents:
        if doc.strip():

            for match in re.finditer(PAT, doc):
                chunk = match.group()
                chunk_ids = [byte_val + num_special for byte_val in chunk.encode("utf-8")]
                all_ids.append(chunk_ids)

            # text_chunks = re.findall(PAT, doc)
            #
            # # Encode to ids with offset
            # ids = [[byte_val + num_special for byte_val in ch.encode("utf-8")]
            #        for ch in text_chunks]
            # all_ids.extend(ids)
    return all_ids


def trainingBPE(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    start_time = timer()
    PAT = re.compile( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    num_special = len(special_tokens)
    # for match in re.finditer(pattern, text):
    # text = "The quick é brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once."

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        # for start, end in zip(boundaries[:-1], boundaries[1:]):
        #     f.seek(start)
        #     chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
    print("opened file ", timer() - start_time)
    chunk_args = [(start, end, input_path, PAT, num_special, special_tokens)
                  for start, end in zip(boundaries[:-1], boundaries[1:])]
    print("split chunks ", timer() - start_time)
    with Pool(num_processes) as pool:
        chunk_results = pool.map(process_chunk, chunk_args)
    print("pool map done ", timer() - start_time)
    # Flatten results into single ids list
    ids = []
    for chunk_ids in chunk_results:
        ids.extend(chunk_ids)
    print("ids extend ", timer() - start_time)

    # ids = [[byte_val + num_special for byte_val in list(ch.encode("utf-8"))] for ch in text_chunks]
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = {} #(int, int) -> int  you had othervise previously newid -> (pair (int, int))
    merges_list = []
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')

    for i in range(256):
        vocab[num_special + i] = bytes([i])

    stats = {}
    for chunk_ids in ids:
        stats = get_stats(chunk_ids, stats)

    #count of words
    # word_frequency = Counter(ids)
    word_frequency = Counter(tuple(word) for word in ids)

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

# trainingBPE("/Users/anko/Documents/Study/cs336-assignment1/tests/fixtures/corpus.en", 500, [] )
