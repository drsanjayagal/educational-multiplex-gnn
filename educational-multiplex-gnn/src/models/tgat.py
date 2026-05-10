import torch
import torch.nn as nn
import torch.nn.functional as F

class TGAT(nn.Module):
def init(self, node_feat_dim, hidden_dim=64, num_heads=4, num_layers=2,
temporal_encoding_dim=16, dropout=0.3, num_hops=2):
super().init()
self.node_embedding = nn.Linear(node_feat_dim, hidden_dim)
self.temporal_proj = nn.Linear(temporal_encoding_dim, hidden_dim)
self.attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)])
self.predictor = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
self.dropout = nn.Dropout(dropout)
def forward(self, node_feat, temporal_enc, edge_index, return_embeds=False):
h = self.node_embedding(node_feat) + self.temporal_proj(temporal_enc)
h = h.unsqueeze(0) # batch size 1
h, _ = self.attn(h, h, h)
for layer in self.layers:
h = F.elu(layer(h))
h = h.squeeze(0)
if return_embeds:
return h
src, dst = edge_index
edge_emb = torch.cat([h[src], h[dst]], dim=-1)
return torch.sigmoid(self.predictor(edge_emb)).squeeze(-1)
