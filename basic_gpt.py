import torch
from torch import nn
from torch.nn import functional as F

n_embd = 384
n_head = 6
n_layer = 6
head_s = n_embd//n_head
# head_s = 32
dropout = 0.2

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head, block_size):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        self.sa = MultiHeadAttention(num_heads=n_head,block_size=block_size, head_size_ = head_s)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size_, block_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadSelfAttention(n_embd=n_embd, head_size=head_size_, block_size=block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size_ * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class SingleHeadSelfAttention(nn.Module):
    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) 
        weights = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        weights = F.softmax(weights, dim=-1) # (B, T, T)
        weights = self.dropout(weights)
        out = weights @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out
    
class VisionGPTModel(nn.Module):
    def __init__(self, vocab_size, device, block_size):
        super().__init__()
        self.device = device
        self.block_size = block_size
        print("vocab_size ", vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, block_size=block_size) for _ in range(n_layer)])
        # self.sa = MultiHeadAttention(num_heads=n_head,block_size=block_size, head_size_ = head_s)
        # self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        # self.ffwd = FeedFoward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)    

    def forward(self, inputs, targets=None):
        B, T = inputs.shape
        input_tensor_clamped = torch.clamp(inputs, 0, self.token_embedding_table.num_embeddings - 1)
        tok_emb = self.token_embedding_table(input_tensor_clamped) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C))) # (T,C)
        # x = self.sa(x)
        x = self.blocks(x)
        # x = self.ffwd(x)
        # x = self.blocks(x) # (B,T,C)
        # x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
       
        if targets is None:
            loss = None
        else:    
            B, T, C = logits.shape  #4,256, 29
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss =None
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # loss = F.cross_entropy(logits, targets)
            # print("...loss.......")
            # print(loss)
            
        return logits, loss

    
    def generate(self, inputs, max_token_len):
        for _ in range(max_token_len):
            idx_cond = inputs[:, -self.block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probability = F.softmax(logits, dim=1)
            inputs_next = torch.multinomial(probability, num_samples=1)
            inputs = torch.cat((inputs, inputs_next), dim=1)
        return  inputs    