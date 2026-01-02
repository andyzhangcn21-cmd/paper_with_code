import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LinearMultiHeadAttention(nn.Module):
    """Linear multi-head attention for linear Transformers."""
    
    def __init__(self, d_model, n_heads, d_k=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        if d_k is None:
            d_k = d_model // n_heads
        
        self.d_k = d_k
        
        # Linear projections (no softmax, just linear)
        self.W_q = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_heads * d_k, bias=False)
        self.W_o = nn.Linear(n_heads * d_k, d_model, bias=False)
        
        # Initialize with identity-like matrices for stable training
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with small weights for stable training
        nn.init.normal_(self.W_q.weight, mean=0, std=0.02)
        nn.init.normal_(self.W_k.weight, mean=0, std=0.02)
        nn.init.normal_(self.W_v.weight, mean=0, std=0.02)
        nn.init.normal_(self.W_o.weight, mean=0, std=0.02)
    
    def forward(self, x):
        """Forward pass of linear attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor of same shape
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # [batch, heads, seq_len, d_k]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Linear attention: A = QK^T / sqrt(d_k) (no softmax)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply attention to values
        attn_output = torch.matmul(scores, v)
        
        # Concatenate heads and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_heads * self.d_k)
        
        output = self.W_o(attn_output)
        
        return output


class LinearTransformerLayer(nn.Module):
    """Single layer linear Transformer for ICL regression."""
    
    def __init__(self, d_model, n_heads, d_k=None):
        super().__init__()
        self.attention = LinearMultiHeadAttention(d_model, n_heads, d_k)
        
        # Simple residual connection
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        """Forward pass through Transformer layer.
        
        Args:
            x: Input sequence [batch_size, seq_len, d_model]
        
        Returns:
            Output sequence [batch_size, seq_len, d_model]
        """
        # Linear attention with residual connection
        attn_output = self.attention(x)
        output = self.norm(x + attn_output)
        
        return output


class LinearTransformerICL(nn.Module):
    """Linear Transformer for in-context learning of linear regression."""
    
    def __init__(self, d_model, n_heads, d_k=None, output_dim=1):
        super().__init__()
        self.transformer = LinearTransformerLayer(d_model, n_heads, d_k)
        
        # Final projection to output (predicts y from x)
        # We use the last token's representation for prediction
        self.output_proj = nn.Linear(d_model, output_dim)
        
        # Learnable token embeddings for examples and query
        self.example_embedding = nn.Parameter(torch.randn(1, d_model))
        self.query_embedding = nn.Parameter(torch.randn(1, d_model))
    
    def forward(self, X_context, y_context, x_query):
        """Make prediction given context examples and query.
        
        Args:
            X_context: Context inputs [batch_size, N, d]
            y_context: Context outputs [batch_size, N, 1]
            x_query: Query input [batch_size, d]
        
        Returns:
            Predicted y for query [batch_size, 1]
        """
        batch_size, N, d = X_context.shape
        
        # Create sequence: [example_1, ..., example_N, query]
        examples = torch.cat([X_context, y_context], dim=-1)  # [batch, N, d+1]
        
        # Add learned embeddings
        example_emb = self.example_embedding.expand(batch_size, N, -1)
        examples_embedded = examples + example_emb
        
        query_emb = self.query_embedding.expand(batch_size, 1, -1)
        query_expanded = x_query.unsqueeze(1)  # [batch, 1, d]
        # Pad query to match embedding dimension
        query_padded = torch.cat([
            query_expanded, 
            torch.zeros(batch_size, 1, 1, device=x_query.device)
        ], dim=-1)
        query_embedded = query_padded + query_emb
        
        # Concatenate context and query
        sequence = torch.cat([examples_embedded, query_embedded], dim=1)
        
        # Pass through transformer
        transformed = self.transformer(sequence)
        
        # Use last token (query) representation for prediction
        query_representation = transformed[:, -1, :]
        
        # Predict y
        y_pred = self.output_proj(query_representation)
        
        return y_pred