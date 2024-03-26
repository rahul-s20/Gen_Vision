import torch
from train  import decode, context_len, vocab_size
from basic_gpt import VisionGPTModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = VisionGPTModel(vocab_size=vocab_size, device=device, block_size=context_len)
checkpoint = torch.load('saved_models/4900_checkpoint.tar')

state_dict = checkpoint['model']
model.load_state_dict(state_dict)
m = model.to(device)


inp = torch.zeros((1, 1), dtype=torch.long)
inp = inp.to(device)
out = decode(model.generate(inp, 500)[0].tolist())

# for i in out:
print(out)
# print(decode(model.generate(inp, 10000)[0].tolist()))