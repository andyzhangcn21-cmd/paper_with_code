# models/minimal_transformer.py

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)  # even index
        pe[:, 1::2] = torch.cos(position * div_term)  # odd index

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
  
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class MinimalTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model)

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,  #  [batch, seq, dim]
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        """
        prompt: [batch, n_context + 1, input_dim]
        return: [batch]  (每个 batch 一个标量预测)
        """
        # [batch, seq, input_dim] → [batch, seq, d_model]
        x = self.input_proj(prompt)

        x = self.pos_encoding(x)

        # context: [batch, n, d_model]
        # query:   [batch, 1, d_model]
        context = x[:, :-1, :]
        query = x[:, -1:, :]

        attn_output, _ = self.attention(
            query,   # query
            context, # key
            context, # value
        )  # attn_output: [batch, 1, d_model]

        out = self.mlp(attn_output)  # [batch, 1, 1]
        
        return out.squeeze(-1).squeeze(-1)

class TheoryICLTransformer(nn.Module):
    """Theory-aligned ICL transformer.

    Q/K depend ONLY on x, V is linear in y, output is linear.
    This makes the predictor an explicit kernel smoother:
        y_hat = b + s * sum_i softmax(q·k_i / sqrt(d)) * y_i
    """

    def __init__(self, d_model: int = 64, max_len: int = 512):
        super().__init__()
        self.d_model = int(d_model)

        self.x_proj = nn.Linear(1, d_model, bias=True)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        self.y_to_v = nn.Linear(1, d_model, bias=False)
        self.readout = nn.Linear(d_model, 1, bias=True)

    def forward(self, prompt: torch.Tensor, return_attn: bool = False):
        x = prompt[..., 0:1]  # [B, L, 1]
        y = prompt[..., 1:2]  # [B, L, 1]

        h = self.x_proj(x)
        h = self.pos_encoding(h)

        context_h = h[:, :-1, :]
        query_h = h[:, -1:, :]

        q = self.q_proj(query_h)        # [B, 1, d]
        k = self.k_proj(context_h)      # [B, n, d]
        v = self.y_to_v(y[:, :-1, :])   # [B, n, d]

        d = q.size(-1)
        logits = (q @ k.transpose(1, 2)) / (d ** 0.5)  # [B, 1, n]
        attn_w = torch.softmax(logits, dim=-1)         # [B, 1, n]
        attn_out = attn_w @ v                          # [B, 1, d]

        y_hat = self.readout(attn_out).squeeze(-1).squeeze(-1)
        if return_attn:
            return y_hat, attn_out, attn_w
        return y_hat
