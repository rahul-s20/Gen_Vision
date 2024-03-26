import torch
from torch import nn
from torch.nn import functional as F
from basic_gpt import VisionGPTModel
import os



corpus = open('data/mini_sekspiere.txt', 'r').read()
uniques = sorted(list(set([i for i in corpus])))
ctoi = {ch: idx for idx, ch in enumerate(uniques)}
itoc = {v:k for k, v in ctoi.items()}

encode = lambda s: [ctoi[i] for i in s]
decode = lambda d: ''.join([itoc[j] for j in d])

data = torch.tensor(encode(corpus), dtype=torch.long)

context_len = 256 #256
batch_size = 64
vocab_size = len(uniques)
epochs = 5000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'



def get_batch():
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y



if __name__ == "__main__()":

    model = VisionGPTModel(vocab_size=vocab_size, device=device, block_size=context_len)
    m = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("starts...............")
    directory = 'saved_models'
    for steps in range(epochs):
        x_, y_  = get_batch()
        logits, loss = model(inputs=x_, targets=y_)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(f"epochs ==> {steps}")
        if steps % 100 == 0:
            print(f"epochs ==> {steps} loss ====> ", loss.item())
            torch.save({
                    'iteration': steps,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'loss': loss,
                    'c_voc': ctoi,
                    'i_voc': itoc,
                }, os.path.join(directory, f'{steps}_checkpoint_basic.tar'))
# inp = torch.zeros((1, 1), dtype=torch.long)
# inp = inp.to(device)
# print(decode(model.generate(inp, 100)[0].tolist()))

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))