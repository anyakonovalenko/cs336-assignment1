import regex as re


def get_stats(ids):
    counts = {}
    proceed = True
    for i in range(len(ids) - 1):
        key_two = (ids[i], ids[i + 1])
        if key_two in counts:
            counts[key_two] += 1
        else:
            counts[key_two] = 1
    return counts

def merge_bpe(ids, pair, idx):

    newids = [] # update text in token with a new pair
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    if i == len(ids) - 1:
        newids.append(ids[i])

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
    print(tokens)
    return tokens

def pre_tokenization():
    PAT =  r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # for match in re.finditer(pattern, text):

    #2 in text page 7 is not clear the difference between original BPE


    text = "The quick é brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet at least once."
    tokens = text.encode("utf-8")
    tokens = list(map(int, tokens))
    vocab_size = 259
    num_merges = vocab_size - 256
    ids = list(tokens)
    merges = {} #(int, int) -> int  you had othervise previously newid -> (pair (int, int))
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=lambda i: stats[i])  # find pair
        iter_num = 256 + i
        ids = merge_bpe(ids, pair, iter_num)
        merges[pair] = iter_num
    encode('werwer jkdljsf the quickl', merges)
    # print(bytes([128]))
    # print(merges)



    # return re.findall(PAT, "some text here.UHUH")
    return 0






print(pre_tokenization())