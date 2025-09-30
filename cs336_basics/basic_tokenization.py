import regex as re


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


def trainingBPE(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    PAT = re.compile( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # for match in re.finditer(pattern, text):
    text = "The quick é brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once."
    # with open(input_path, 'r', encoding='utf-8') as f:
    #     text = f.read()
    text_chunks = re.findall(PAT, text)
    num_special = len(special_tokens)
    # ids = [[symbol for symbol in ch] for ch in text_chunks]
    ids = [[byte_val + num_special for byte_val in list(ch.encode("utf-8"))] for ch in text_chunks]
    # vocab_size = 259
    num_merges = vocab_size - 256 - len(special_tokens)
    merges = {} #(int, int) -> int  you had othervise previously newid -> (pair (int, int))
    merges_list = []
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode('utf-8')

    for i in range(256):
        vocab[num_special + i] = bytes([i])

    for merge_id in range(num_merges):
        stats = {}
        for chunk_ids in ids:
            stats = get_stats(chunk_ids, stats)

        # Use vocab to get byte representation for lexicographic comparison
        pair = max(stats, key=lambda p: (stats[p], (vocab[p[0]]), vocab[p[1]]))

        iter_num = num_special+ 256 + merge_id
        vocab[iter_num] = vocab[pair[0]] + vocab[pair[1]]  # Add new token to vocab
        ids = merge_bpe(ids, pair, iter_num)
        merges[pair] = iter_num

    for (m1, m2), value in merges.items():
        merges_list.append((vocab[m1], vocab[m2]))


    return vocab, merges_list

trainingBPE('ff', 300, ['fdf'])