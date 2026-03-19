import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

device = torch.device("cpu")

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        output = self.combine_heads(attn_output)
        return self.W_o(output)

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm2 = AddNorm(d_model)

    def forward(self, x, src_mask=None):
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.add_norm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.add_norm2(x, ffn_output)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm1 = AddNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.add_norm2 = AddNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff)
        self.add_norm3 = AddNorm(d_model)

    def forward(self, y, memory, tgt_mask=None, src_mask=None):
        masked_attn_output = self.masked_self_attn(y, y, y, tgt_mask)
        y = self.add_norm1(y, masked_attn_output)
        cross_attn_output = self.cross_attn(y, memory, memory, src_mask)
        y = self.add_norm2(y, cross_attn_output)
        ffn_output = self.ffn(y)
        y = self.add_norm3(y, ffn_output)
        return y

class TransformerFromScratch(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=64, num_heads=4, d_ff=128, num_layers=2, max_len=100):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        y = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        y = self.positional_encoding(y)
        for layer in self.decoder_layers:
            y = layer(y, memory, tgt_mask, src_mask)
        return y

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        y = self.decode(tgt, memory, tgt_mask, src_mask)
        logits = self.output_layer(y)
        return logits

def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

def create_causal_mask(size):
    return torch.tril(torch.ones(size, size, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

def create_decoder_mask(tgt, pad_idx):
    pad_mask = create_padding_mask(tgt, pad_idx)
    causal_mask = create_causal_mask(tgt.size(1)).to(tgt.device)
    return pad_mask & causal_mask

src_vocab = {
    "<PAD>": 0,
    "Thinking": 1,
    "Machines": 2
}

tgt_vocab = {
    "<PAD>": 0,
    "<START>": 1,
    "<EOS>": 2,
    "Maquinas": 3,
    "Pensantes": 4
}

src_id_to_token = {idx: token for token, idx in src_vocab.items()}
tgt_id_to_token = {idx: token for token, idx in tgt_vocab.items()}

src_pad_idx = src_vocab["<PAD>"]
tgt_pad_idx = tgt_vocab["<PAD>"]

src_sentence = ["Thinking", "Machines"]
tgt_input_sentence = ["<START>", "Maquinas", "Pensantes"]
tgt_output_sentence = ["Maquinas", "Pensantes", "<EOS>"]

src_tensor = torch.tensor([[src_vocab[token] for token in src_sentence]], dtype=torch.long, device=device)
tgt_input_tensor = torch.tensor([[tgt_vocab[token] for token in tgt_input_sentence]], dtype=torch.long, device=device)
tgt_output_tensor = torch.tensor([[tgt_vocab[token] for token in tgt_output_sentence]], dtype=torch.long, device=device)

model = TransformerFromScratch(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    d_model=64,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    max_len=20
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

model.train()
for epoch in range(2000):
    src_mask = create_padding_mask(src_tensor, src_pad_idx).to(device)
    tgt_mask = create_decoder_mask(tgt_input_tensor, tgt_pad_idx).to(device)

    logits = model(src_tensor, tgt_input_tensor, src_mask, tgt_mask)
    loss = criterion(logits.view(-1, logits.size(-1)), tgt_output_tensor.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if loss.item() < 0.0005:
        break

model.eval()

with torch.no_grad():
    src_mask = create_padding_mask(src_tensor, src_pad_idx).to(device)
    memory = model.encode(src_tensor, src_mask)

    generated = [tgt_vocab["<START>"]]
    max_len = 10

    while len(generated) < max_len:
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_mask = create_decoder_mask(tgt_tensor, tgt_pad_idx).to(device)

        decoder_output = model.decode(tgt_tensor, memory, tgt_mask, src_mask)
        logits = model.output_layer(decoder_output)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()

        if next_token_id == tgt_vocab["<EOS>"]:
            break

        generated.append(next_token_id)

generated_tokens = [tgt_id_to_token[idx] for idx in generated[1:]]
print("Frase de entrada:", " ".join(src_sentence))
print("Frase gerada:", " ".join(generated_tokens))
print("Loss final:", round(loss.item(), 6))