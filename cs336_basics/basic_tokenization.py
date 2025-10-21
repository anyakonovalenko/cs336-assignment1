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


def trainingBPE(input_path, vocab_size, special_tokens) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    PAT = re.compile( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    # for match in re.finditer(pattern, text):
    # text = "The quick é brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once."
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    text_chunks = re.findall(PAT, text)
    num_special = len(special_tokens)
    ids = [[byte_val + num_special for byte_val in list(ch.encode("utf-8"))] for ch in text_chunks]
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

    for merge_id in range(num_merges):
        new_ids = []
        # Use vocab to get byte representation for lexicographic comparison
        pair = max(stats, key=lambda p: (stats[p], (vocab[p[0]]), vocab[p[1]]))
        iter_num = num_special + 256 + merge_id
        vocab[iter_num] = vocab[pair[0]] + vocab[pair[1]]  # Add new token to vocab

        for chunk_id in ids:
            i = 0
            new_chunk = []
            while i < len(chunk_id):
                if i < len(chunk_id) - 1 and chunk_id[i] == pair[0] and chunk_id[i + 1] == pair[1]:
                    if i > 0:
                        stats[(chunk_id[i-1], pair[0])] = stats.get((chunk_id[i-1], pair[0]), 0) - 1
                        stats[(chunk_id[i-1], iter_num)] = stats.get((chunk_id[i-1], iter_num), 0) + 1

                    if i + 2 < len(chunk_id):
                        stats[(pair[1], chunk_id[i + 2])] = stats.get((pair[1], chunk_id[i + 2]), 0) - 1
                        stats[(iter_num, chunk_id[i+2])] = stats.get((iter_num, chunk_id[i+2]), 0) + 1

                    stats[(pair[0], pair[1])] = stats.get((pair[0], pair[1]), 0) - 1
                    if stats[(pair[0], pair[1])] == 0:
                        del stats[(pair[0], pair[1])]

                    new_chunk.append(iter_num)
                    i += 2
                else:
                    new_chunk.append(chunk_id[i]) #if i the lst element (len - 1)
                    i += 1
            new_ids.append(new_chunk)

        ids = new_ids
        merges[pair] = iter_num

    for (m1, m2), value in merges.items():
        merges_list.append((vocab[m1], vocab[m2]))


    return vocab, merges_list

# trainingBPE('ff', 300, ['fdf'])