def ctoi(uniques):
    return {ch: idx for idx, ch in enumerate(uniques)}

def itoc(ctoi_dict):
    return {v:k for k, v in ctoi_dict.items()}

def extracts_uniques(corpus):
    uniques = sorted(list(set([i for i in corpus])))
    return uniques

def encode(ctoi_dict, text_list):
    return [ctoi_dict[i] for i in text_list]

def decode(itoc_dict, token_list):
    return ''.join([itoc_dict[j] for j in token_list])
