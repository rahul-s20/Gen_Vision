import tiktoken
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from basic_gpt import VisionGPTModel
import os
from collections import OrderedDict

enc = tiktoken.get_encoding("gpt2")

data = pd.read_csv('data/chatbot_dataset.txt', sep='\t', names=['questions', 'answers'])
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
context_len = 84 #256
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 0.0001
epochs = 6000
eval_iters = 200
eval_interval = 500
# device = 'cpu'
@torch.no_grad()
def estimate_loss(inp_tensor, out_tensor, start_idx, end_idx):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(inp_tensor, out_tensor, start_idx, end_idx)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def padding(data_lst):
    return [i+ [0] * (context_len - len(i)) for i in data_lst]

def encode_inp_op(data):
    _x = [enc.encode_ordinary(i)for i in data.questions.to_list()]
    _y = [enc.encode_ordinary(j) for j in data.answers.to_list()]
    # for h in _y:
    #     if len(h) > context_len:
    #         print(len(h) )
    x = torch.tensor(padding(_x), dtype=torch.long)
    y = torch.tensor(padding(_y), dtype=torch.long)
    return x, y

def get_batch(inp_tensor, out_tensor, start_idx, end_idx):
    # i = torch.stack([inp_tensor[start_idx:end_idx]])
    # o = torch.stack([out_tensor[start_idx:end_idx]])
    i = inp_tensor[start_idx:end_idx]
    o = out_tensor[start_idx:end_idx]
    return i.to(device), o.to(device)

def get_unique_words(data):
    lis = []
    x = data.questions.to_list()
    y = data.answers.to_list()
    full = x + y
    for m in full:
        lis.extend(m.split())

    return list(OrderedDict.fromkeys(lis))


i_tensor, o_tensor = encode_inp_op(data)
uniques = get_unique_words(data)      
# X, Y  = get_batch(i_tensor, o_tensor, 0, 32)

# print(i_tensor[:2].to(device))
# print("==================")
# print(torch.stack([i_tensor[:2]]))

if __name__ == '__main__':
    model = VisionGPTModel(vocab_size=50304, device=device, block_size=context_len)
    m = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    print("Model Parameters ========================> ")
    print(model.parameters())
    directory = 'saved_models'
    start = 0
    end = batch_size
    for steps in range(epochs):
        if end >= len(train_data):
            start = 0
            end = batch_size
        # if steps % eval_interval == 0 or steps == epochs - 1:
        #     losses = estimate_loss()
        #     print(f"step {steps}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        X, Y  = get_batch(i_tensor, o_tensor, start, end)
        start = end
        end += batch_size 

        logits, loss = model(inputs=X, targets=Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()    

        if steps % eval_interval == 0 or steps == epochs - 1:
            print(f"step {steps}: train loss {loss.item()}")
            torch.save({
                    'iteration': steps,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(directory, f'{steps}_chatbot_checkpoint_basic.tar'))   