import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from utils import normalize_adj2, sparse_mx_to_torch_sparse_tensor


class WeightedGATConv(GATConv):
    """
    Custom Graph Attention Convolution layer that incorporates edge weights into the attention mechanism.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        super(WeightedGATConv, self).__init__(
            in_channels,
            out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        size: Optional[tuple] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor:
        if edge_weight is not None:
            # Convert edge_weight to shape [E, 1] as edge_attr
            edge_attr = edge_weight.view(-1, 1)
            return super().forward(
                x, edge_index, edge_attr=edge_attr,
                size=size, return_attention_weights=return_attention_weights
            )
        else:
            return super().forward(
                x, edge_index, edge_attr=None,
                size=size, return_attention_weights=return_attention_weights
            )


class SineCosinePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:T, :].unsqueeze(0)


class TransformerTemporalEncoder(nn.Module):
    def __init__(self,
                 d_model=32,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=128,
                 dropout=0.1,
                 max_time_len=5000):
        super().__init__()
        self.d_model = d_model
        self.input_projection = None
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.pos_encoder = SineCosinePositionalEncoding(d_model, max_len=max_time_len)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B, T_in, m, C = X.shape
        X_reshape = X.permute(0, 2, 1, 3).contiguous()
        X_reshape = X_reshape.view(B*m, T_in, C)
        
        if self.input_projection is None:
            self.input_projection = nn.Linear(C, self.d_model, bias=True).to(X.device)
        
        x = self.input_projection(X_reshape)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        x = x.view(B, m, self.d_model)
        x = self.layer_norm(x)
        return x


class SpatiotemporalTransformerGAT(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.m = data.m
        self.w = args.window
        self.d_model = args.d_model
        self.dropout = args.dropout
        self.h = args.horizon
        
        # Get normalized adjacency matrix and edge weights
        if args.cuda:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense().cuda()
        else:
            self.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj2(data.orig_adj.cpu().numpy())).to_dense()
        
        # Extract edge weights from adjacency matrix
        self.edge_index = self.adj.nonzero(as_tuple=False).t().contiguous()
        self.edge_weight = self.adj[self.edge_index[0], self.edge_index[1]]

        # Temporal encoder
        self.transformer = TransformerTemporalEncoder(
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_transformer_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )

        # Weighted GAT layer
        self.gat = WeightedGATConv(
            in_channels=args.d_model,
            out_channels=args.hidden_dim_gnn // args.gat_heads,
            heads=args.gat_heads,
            dropout=args.dropout + 0.2,
            concat=True,
            add_self_loops=True,
            bias=False
        )
        self.ln = nn.LayerNorm(args.hidden_dim_gnn)
        
        # Skip connection projection
        self.skip_projection = nn.Linear(1, args.hidden_dim_gnn)
        self.skip_norm = nn.LayerNorm(args.hidden_dim_gnn)
        
        # Output projection with residual connections
        self.output_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.hidden_dim_gnn, args.hidden_dim_gnn),
                nn.LayerNorm(args.hidden_dim_gnn),
                nn.GELU(),
                nn.Dropout(args.dropout + 0.1),
                nn.Linear(args.hidden_dim_gnn, args.hidden_dim_gnn)
            ) for _ in range(2)  # Two residual blocks
        ])
        self.final_projection = nn.Linear(args.hidden_dim_gnn, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, X: torch.Tensor) -> tuple:
        batch_size = X.size(0)
        
        # Add feature dimension
        X = X.unsqueeze(-1)
        
        # Skip connection from input
        x_skip = X[:, -1]  # Use last time step
        x_skip = self.skip_projection(x_skip)
        x_skip = self.skip_norm(x_skip)
        
        # Temporal encoding
        node_emb = self.transformer(X)
        
        outputs = []
        for b in range(batch_size):
            x = node_emb[b]
            # Use weighted GAT with edge weights
            x, _ = self.gat(x, self.edge_index, edge_weight=self.edge_weight, return_attention_weights=True)
            x = self.ln(x)
            x = F.gelu(x)
            outputs.append(x.unsqueeze(0))
        
        x = torch.cat(outputs, dim=0)
        
        # Combine with skip connection (more emphasis on spatial features)
        x = 0.8 * x + 0.2 * x_skip
        
        # Multi-step prediction
        predictions = []
        x_in = x
        
        # Iteratively predict each step
        for step in range(self.h):
            # Project through residual blocks
            x_step = x_in.reshape(batch_size * self.m, -1)
            for proj in self.output_projection:
                x_step = proj(x_step) + x_step  # Residual connection
            x_step = self.final_projection(x_step)  # [batch*nodes, 1]
            x_step = x_step.reshape(batch_size, self.m)  # [batch, nodes]
            predictions.append(x_step)
            
            # Update input for next step (more emphasis on current prediction)
            x_in = 0.6 * x_in + 0.4 * x_step.unsqueeze(-1).expand(-1, -1, x_in.size(-1))
        
        # Stack predictions and match target shape [batch, nodes, horizon]
        x = torch.stack(predictions, dim=-1)  # [batch, nodes, horizon]
        
        return x, None
