import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn import Parameter
from torch_geometric.nn import GCNConv
import logging
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        pe = self.pe[:, :x.size(1), :]
        x = x + pe
        return self.dropout(x)
class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(MultiScaleTemporalBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        branch_channels = out_channels // 3
        remainder = out_channels % 3
        self.branch1_channels = branch_channels + (1 if remainder > 0 else 0)
        self.branch2_channels = branch_channels + (1 if remainder > 1 else 0)
        self.branch3_channels = branch_channels
        self.branch1 = nn.Linear(in_channels, self.branch1_channels)
        self.branch2 = nn.Linear(in_channels, self.branch2_channels)
        self.branch3 = nn.Linear(in_channels, self.branch3_channels)
        self.fusion = nn.Linear(self.branch1_channels + self.branch2_channels + self.branch3_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x1 = F.relu(self.branch1(x))
        x2 = F.relu(self.branch2(x))
        x3 = F.relu(self.branch3(x))
        x_cat = torch.cat([x1, x2, x3], dim=1)
        out = F.relu(self.fusion(x_cat))
        out = self.bn(out)
        out = self.dropout(out)
        return out
class EnhancedTemporalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.input_transform = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=0.1
        )
        self.feature_extract = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        batch_size = x.size(0)
        x = self.input_transform(x)
        if x.dim() == 4:
            x = x.view(batch_size, -1, self.hidden_dim)
        x = self.pos_encoder(x)
        q = self.query(x).permute(1, 0, 2)  
        k = self.key(x).permute(1, 0, 2)
        v = self.value(x).permute(1, 0, 2)
        attended, attention_weights = self.attention(q, k, v)
        attended = attended.permute(1, 0, 2)  
        attended = self.norm1(x + self.dropout(attended))
        features = self.feature_extract(attended)
        gate_input = torch.cat([attended, features], dim=-1)
        gate = self.gate(gate_input)
        gated_features = features * gate
        out = self.norm2(attended + self.dropout(gated_features))
        return out, attention_weights
class GCA(nn.Module):
    def __init__(self, in_channels, num_heads=8, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        assert self.head_dim * num_heads == in_channels, "in_channels must be divisible by num_heads"
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(in_channels, in_channels)
        self.out_proj = nn.Linear(in_channels, in_channels)
        self.layer_norm1 = nn.LayerNorm(in_channels)
        self.layer_norm2 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        batch_size = x.size(0)
        residual = x
        x = self.layer_norm1(x)
        q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        k = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        v = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        src, dst = edge_index
        valid_mask = (src < batch_size) & (dst < batch_size)
        if not valid_mask.all():
            logging.warning(f"found invalid edge index: src_max={src.max()}, dst_max={dst.max()}, batch_size={batch_size}")
            src = src[valid_mask]
            dst = dst[valid_mask]
            if len(src) == 0:
                logging.warning("all edge indexes are invalid, return original feature")
                return residual
        q_dst = q[dst]  
        k_src = k[src]  
        attn_scores = torch.sum(q_dst * k_src, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=0).unsqueeze(-1)
        v_src = v[src]
        weighted_values = v_src * attn_weights
        aggregated = torch.zeros_like(v)
        for i in range(dst.size(0)):
            aggregated[dst[i]] += weighted_values[i]
        output = self.out_proj(aggregated.view(batch_size, -1))
        output = self.dropout(output)
        x = residual + output
        residual = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        output = residual + x
        return output
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.transform = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.edge_attention = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        self.node_attention = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x, edge_index):
        h = self.transform(x)
        row, col = edge_index
        edge_features = torch.cat([h[row], h[col]], dim=1)
        edge_weights = self.edge_attention(edge_features).squeeze(-1)
        node_weights = self.node_attention(h).squeeze(-1)
        aggr = torch.zeros_like(h)
        for i in range(edge_index.size(1)):
            src, dst = row[i], col[i]
            aggr[dst] += edge_weights[i] * h[src]
        gate_input = torch.cat([h, aggr], dim=-1)
        update_gate = self.update_gate(gate_input)
        h_new = update_gate * aggr + (1 - update_gate) * h
        h_new = self.norm(h_new)
        return h_new
class EnhancedSparseLoss(nn.Module):
    def __init__(self, pos_weight=None, focal_gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.focal_gamma = focal_gamma
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)
        bce_loss = self.bce_loss(pred, target)
        probs = torch.sigmoid(pred)
        pt = torch.where(target == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss = focal_weight * bce_loss
        return loss.mean()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.float().view(-1)
        bce_loss = self.bce_loss(inputs, targets)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = self.alpha * focal_weight * bce_loss
        return focal_loss.mean()
class AdaptiveDiffusionLoss(torch.nn.Module):
    def __init__(self, pos_weight=1.0, temperature=0.1, alpha=0.5, gamma=2.0):
        super(AdaptiveDiffusionLoss, self).__init__()
        self.pos_weight = pos_weight
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self.focal = FocalLoss(alpha=pos_weight, gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight if isinstance(pos_weight, torch.Tensor) else torch.tensor([pos_weight]))
    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.float().view(-1)
        focal_loss = self.focal(pred, target)
        bce_loss = self.bce(pred, target)
        pred_prob = torch.sigmoid(pred)
        diff = torch.abs(pred_prob - target)
        diff = diff / self.temperature
        diffusion_loss = torch.mean(diff)
        total_loss = (1 - self.alpha) * focal_loss + self.alpha * diffusion_loss
        return total_loss
class GCALayer(nn.Module):
    def __init__(self, in_channels, out_channels, topology_channels, heads=1,
                 concat=True, negative_slope=0.2, dropout=0.0, bias=True):
        super(GCALayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.topology_channels = topology_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin = nn.Linear(in_channels, heads * out_channels, bias=bias)
        self.att_node = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_topology = Parameter(torch.Tensor(1, heads, out_channels))
        self.lin_topology = nn.Linear(topology_channels, heads * out_channels, bias=bias)
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()
    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin_topology.reset_parameters()
        nn.init.xavier_uniform_(self.att_node)
        nn.init.xavier_uniform_(self.att_topology)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    def forward(self, x, edge_index, topology_features):
        batch_size, num_nodes, _ = x.size()
        if edge_index.dim() > 2:
            edge_index = edge_index.squeeze()
        if edge_index.size(0) != 2:
            edge_index = edge_index.t()  
        row, col = edge_index[0], edge_index[1]
        if x.size(-1) != self.in_channels:
            x = x[:, :self.in_channels]
        x = self.lin(x).view(batch_size, num_nodes, self.heads, self.out_channels)
        x_i, x_j = x[:, row], x[:, col]
        topology = self.lin_topology(topology_features).view(batch_size, num_nodes, self.heads, self.out_channels)
        topo_i, topo_j = topology[:, row], topology[:, col]
        alpha_node = (x_j * self.att_node).sum(dim=-1)
        alpha_topology = (topo_j * self.att_topology).sum(dim=-1)
        alpha = F.leaky_relu(alpha_node + alpha_topology, self.negative_slope)
        alpha = F.softmax(alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = torch.zeros(batch_size, num_nodes, self.heads, self.out_channels, device=x.device)
        out.index_add_(1, row, x_j * alpha.unsqueeze(-1))
        if self.concat:
            out = out.view(batch_size, num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=2)
        if self.bias is not None:
            out += self.bias
        self._alpha = alpha
        return out, topology_features
class EnhancedFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=3):
        super(EnhancedFeatureExtractor, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        ))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.1)
            ))
        self.layers.append(nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        ))
        self.skip_connections = nn.ModuleList([
            nn.Linear(in_channels, hidden_channels) if i == 0 else nn.Linear(hidden_channels, hidden_channels)
            for i in range(num_layers)
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            skip = self.skip_connections[i](x)
            x = layer(x) + skip
        return x
class FeatureFusion(nn.Module):
    def __init__(self, temporal_channels, gene_expression_channels, topology_channels, hidden_channels):
        super(FeatureFusion, self).__init__()
        self.temporal_extractor = EnhancedFeatureExtractor(temporal_channels, hidden_channels)
        self.gene_expression_extractor = EnhancedFeatureExtractor(gene_expression_channels, hidden_channels)
        self.topology_extractor = EnhancedFeatureExtractor(topology_channels, hidden_channels)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels * 3),
            nn.Sigmoid()
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, temporal_features, gene_expression_features, topology_features):
        temporal = self.temporal_extractor(temporal_features)
        gene_expression = self.gene_expression_extractor(gene_expression_features)
        topology = self.topology_extractor(topology_features)
        combined = torch.cat([temporal, gene_expression, topology], dim=-1)
        gate = self.fusion_gate(combined)
        fused = combined * gate
        return self.output(fused)
class MultiScaleAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, num_scales=3):
        super(MultiScaleAttention, self).__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_scales)
        ])
        self.attention_heads = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=0.1
            ) for _ in range(num_scales)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_scales, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    def forward(self, x):
        if x.dim() == 4:  
            x = x.squeeze(1)  
        batch_size, num_genes, _ = x.size()
        multi_scale_features = []
        for i in range(self.num_scales):
            scale_x = self.scale_transforms[i](x)  
            scale_x = scale_x.transpose(0, 1)  
            attn_output, _ = self.attention_heads[i](
                scale_x,  
                scale_x,  
                scale_x   
            )
            attn_output = attn_output.transpose(0, 1)
            multi_scale_features.append(attn_output)
        combined = torch.cat(multi_scale_features, dim=-1)  
        fused = self.fusion(combined)  
        output = self.output(fused + x)
        return output
class DeepGeneRegulationNetwork(nn.Module):
    def __init__(self, num_genes, temporal_channels, gene_expression_channels, topology_channels, hidden_channels=256, num_heads=8):
        super(DeepGeneRegulationNetwork, self).__init__()
        self.num_genes = num_genes
        self.hidden_channels = hidden_channels
        self.feature_importance = nn.ModuleDict({
            'temporal': nn.Parameter(torch.ones(temporal_channels)),
            'gene_expression': nn.Parameter(torch.ones(gene_expression_channels)),
            'topology': nn.Parameter(torch.ones(topology_channels))
        })
        self.feature_fusion = FeatureFusion(
            temporal_channels, gene_expression_channels, topology_channels, hidden_channels
        )
        self.multi_scale_attention = MultiScaleAttention(hidden_channels)
        self.gca_layers = nn.ModuleList([
            GCA(hidden_channels, num_heads=num_heads)
            for _ in range(3)  
        ])
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(3)  
        ])
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels // 2, 1)
        )
    def forward(self, temporal_features, gene_expression_features, topology_features, edge_index=None):
        batch_size = temporal_features.size(0)
        temporal_features = temporal_features * F.softmax(self.feature_importance['temporal'], dim=0).view(1, 1, -1)
        gene_expression_features = gene_expression_features * F.softmax(self.feature_importance['gene_expression'], dim=0).view(1, 1, -1)
        topology_features = topology_features * F.softmax(self.feature_importance['topology'], dim=0).view(1, 1, -1)
        x = self.feature_fusion(temporal_features, gene_expression_features, topology_features)
        x = self.multi_scale_attention(x)
        for gca_layer, layer_norm in zip(self.gca_layers, self.layer_norms):
            gca_out = gca_layer(x, edge_index)
            x = layer_norm(x + gca_out)
        predictions = self.predictor(x)
        return predictions
class TopologyFeatureExtractor:
    def __init__(self):
        self.num_features = 3  
    def extract_features(self, edge_index, num_nodes):
        features = torch.zeros((num_nodes, self.num_features))
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        if edge_index.size(1) > 0:  
            adj_matrix[edge_index[0], edge_index[1]] = 1
        degree = adj_matrix.sum(dim=1)
        features[:, 0] = degree
        in_degree = adj_matrix.sum(dim=0)
        features[:, 1] = in_degree
        out_degree = adj_matrix.sum(dim=1)
        features[:, 2] = out_degree
        for i in range(self.num_features):
            col_mean = features[:, i].mean()
            col_std = features[:, i].std()
            if col_std > 0:
                features[:, i] = (features[:, i] - col_mean) / col_std
        return features
class EnhancedGCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.1):
        super(EnhancedGCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.gcn = GCNConv(in_channels, out_channels)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, edge_index):
        residual = x
        x = self.norm1(x)
        x = self.gcn(x, edge_index)
        x = self.dropout(x)
        x = residual + x
        x = self.norm2(x)
        x = self.dropout(x)
        return x