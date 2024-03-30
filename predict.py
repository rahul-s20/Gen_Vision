# import torch
# from train  import decode, context_len, vocab_size
# from basic_gpt import VisionGPTModel

# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model = VisionGPTModel(vocab_size=vocab_size, device=device, block_size=context_len)
# checkpoint = torch.load('saved_models/4900_checkpoint.tar')

# state_dict = checkpoint['model']
# model.load_state_dict(state_dict)
# m = model.to(device)


# inp = torch.zeros((1, 1), dtype=torch.long)
# inp = inp.to(device)
# out = decode(model.generate(inp, 500)[0].tolist())

# for i in out:
# print(out)
# print(decode(model.generate(inp, 10000)[0].tolist()))
##############################################################
import torch
from basic_gpt import VisionGPTModel
import tiktoken


context_len = 84

enc = tiktoken.get_encoding("gpt2")
decode = lambda l: enc.decode(l)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def padding(data_lst):
    return [i+ [0] * (context_len - len(i)) for i in data_lst]

model = VisionGPTModel(vocab_size=50304, device=device, block_size=context_len)
checkpoint = torch.load('saved_models/1500_chatbot_checkpoint_basic.tar')

state_dict = checkpoint['model']
model.load_state_dict(state_dict)
m = model.to(device)


inp = 'where do you go'
enc_input = [enc.encode_ordinary(i)for i in inp.split()]
x_tensor = torch.tensor(enc_input, dtype=torch.long)
x_tensor = x_tensor.to(device)
out = decode(model.generate(x_tensor, context_len)[0].tolist())
print(out)