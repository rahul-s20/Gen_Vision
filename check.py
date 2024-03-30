import tiktoken
import torch
import torch.nn.functional as F

enc = tiktoken.get_encoding("gpt2")

print(enc.encode_ordinary('\n'))
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

# source = torch.rand((3,5))
# # print(source.shape)
# # print(source.shape[0])
# # print(source.shape[1])
# source_pad = F.pad(source, pad=(0, 5, 0, 10 - source.shape[0]))
# print(source_pad)
# # source.shape
# # torch.Size([3, 42])

# # pad_x = torch.zeros((3, 10), device=device, dtype=torch.long)
# # pad_x[:source.size(0), :] = source[1]
# # print(pad_x)