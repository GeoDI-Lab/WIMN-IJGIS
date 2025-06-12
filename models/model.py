import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_undirected


def diag_flatten_function(matrix):
    mask = torch.triu(torch.ones_like(matrix), diagonal=1).bool()
    return torch.masked_select(matrix, mask)


def create_subgraph_from_hub_nodes(node_attr, hub_nodes):
    """Create a subgraph from hub nodes including node attributes."""
    return node_attr[hub_nodes]


class EdgeNodeFusion(nn.Module):
    def __init__(self, dropout_rate):
        super(EdgeNodeFusion, self).__init__()
        self.p = dropout_rate
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, node_attr, edge_index, edge_attr, dg, gcn, cat_ln, adjacency_indices, edge_mask=None):
        edge_attr = F.leaky_relu(dg(edge_attr))
        edge_attr = F.dropout(edge_attr, p=self.p)

        if edge_mask is not None:
            if torch.is_tensor(edge_mask):
                edge_attr = edge_attr[edge_mask]
            else:
                edge_attr = edge_attr[edge_mask[1]]

        edge_index_full, edge_attr_full = to_undirected(edge_index, edge_attr[adjacency_indices.bool()])
        avg_edge_attr = self.global_avg_pool(edge_attr_full.unsqueeze(1)).squeeze(2)

        x = F.leaky_relu(gcn(x=node_attr, edge_index=edge_index_full, edge_weight=avg_edge_attr))
        x = F.dropout(x, p=self.p)

        edge_weight = F.sigmoid(torch.mm(x, x.t()))
        edge_weight = diag_flatten_function(edge_weight).unsqueeze(-1)
        edge_weight = cat_ln(edge_weight)

        edge_attr = edge_weight * edge_attr + edge_attr
        return edge_attr, x


class Encoder(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dims, edge_input_dim, edge_hidden_dims, dropout_rate):
        super(Encoder, self).__init__()
        self.edge_node_fusion = EdgeNodeFusion(dropout_rate)

        # Deep gravity encoders
        self.deep_gravity_enc = nn.ModuleList(
            [nn.Linear(in_features=edge_input_dim if i == 0 else edge_hidden_dims[i-1],
                       out_features=edge_hidden_dims[i]) for i in range(len(edge_hidden_dims))]
        )

        # GCN encoders
        self.gcn_enc = nn.ModuleList(
            [GCNConv(in_channels=node_input_dim if i == 0 else node_hidden_dims[i-1],
                     out_channels=node_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Concatenation linear layers
        self.concat_linear_enc = nn.ModuleList(
            [nn.Linear(in_features=1, out_features=edge_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

    def forward(self, node_input_x, edge_index, interaction_index, edge_attr, interaction_indices_lists):
        edge_features, node_features = node_input_x, edge_attr

        for i in range(len(self.gcn_enc)):
            edge_features, node_features = self.edge_node_fusion(
                node_attr=node_features,
                edge_index=interaction_index,
                edge_attr=edge_features,
                dg=self.deep_gravity_enc[i],
                gcn=self.gcn_enc[i],
                cat_ln=self.concat_linear_enc[i],
                adjacency_indices=interaction_indices_lists[0]
            )

        return edge_features, node_features


class Decoder(nn.Module):
    def __init__(self, node_input_dim, weather_input_dim, node_hidden_dims, edge_input_dim, edge_hidden_dims, dropout_rate):
        super(Decoder, self).__init__()
        self.edge_node_fusion = EdgeNodeFusion(dropout_rate)

        # Deep gravity decoders
        self.deep_gravity_dec_top = nn.ModuleList(
            [nn.Linear(in_features=edge_input_dim if i == 0 else edge_hidden_dims[i-1],
                       out_features=edge_hidden_dims[i]) for i in range(len(edge_hidden_dims))]
        )

        # GCN decoders
        self.gcn_dec_top = nn.ModuleList(
            [GCNConv(in_channels=node_input_dim + weather_input_dim if i == 0 else node_hidden_dims[i-1] + weather_input_dim,
                     out_channels=node_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Concatenation linear layers
        self.concat_linear_dec_top = nn.ModuleList(
            [nn.Linear(in_features=1, out_features=edge_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Deep gravity decoders
        self.deep_gravity_dec_hub = nn.ModuleList(
            [nn.Linear(in_features=edge_input_dim if i == 0 else edge_hidden_dims[i-1],
                       out_features=edge_hidden_dims[i]) for i in range(len(edge_hidden_dims))]
        )

        # GCN decoders
        self.gcn_dec_hub = nn.ModuleList(
            [GCNConv(in_channels=node_input_dim + weather_input_dim if i == 0 else node_hidden_dims[i-1] + weather_input_dim,
                     out_channels=node_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Concatenation linear layers
        self.concat_linear_dec_hub = nn.ModuleList(
            [nn.Linear(in_features=1, out_features=edge_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Deep gravity decoders
        self.deep_gravity_dec = nn.ModuleList(
            [nn.Linear(in_features=edge_input_dim if i == 0 else edge_hidden_dims[i-1],
                       out_features=edge_hidden_dims[i]) for i in range(len(edge_hidden_dims))]
        )

        # GCN decoders
        self.gcn_dec = nn.ModuleList(
            [GCNConv(in_channels=node_input_dim + weather_input_dim if i == 0 else node_hidden_dims[i-1] + weather_input_dim,
                     out_channels=node_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

        # Concatenation linear layers
        self.concat_linear_dec = nn.ModuleList(
            [nn.Linear(in_features=1, out_features=edge_hidden_dims[i]) for i in range(len(node_hidden_dims))]
        )

    def forward(self, node_hidden_z, weather_input_x, edge_hidden_z, adjacency_index, hub_indices_list, adjacency_indices_lists, hub_top_edge_mask):
        weather_x, hub_weather_x, top_weather_x = weather_input_x
        edge_features, node_features = edge_hidden_z, node_hidden_z
        adjacency_index, hub_adjacency_index, top_adjacency_index = adjacency_index
        hub_edge_mask, top_edge_mask = hub_top_edge_mask

        # Process top layer
        top_node_features = create_subgraph_from_hub_nodes(node_attr=node_features, hub_nodes=hub_indices_list[1])
        top_x = torch.cat([top_node_features, top_weather_x], dim=-1)
        for i in range(len(self.gcn_enc_top)):
            top_edge_features, top_node_features = self.edge_node_fusion(
                node_attr=top_x,
                edge_index=top_adjacency_index,
                edge_attr=edge_features,
                dg=self.deep_gravity_dec_top[0],
                gcn=self.gcn_dec_top[0],
                cat_ln=self.concat_linear_dec_top[0],
                adjacency_indices=adjacency_indices_lists[-1],
                edge_mask=[hub_edge_mask, top_edge_mask])
        top_edge_features = self.edge_unpool(selected_edge_attr = top_edge_features, original_num_edges = edge_features.shape[0], selected_edges=hub_edge_mask)


        # Process hub layer
        hub_node_features = create_subgraph_from_hub_nodes(node_attr=node_features, hub_nodes=hub_indices_list[0])
        hub_x = torch.cat([hub_node_features, hub_weather_x], dim=-1)
        for i in range(len(self.gcn_enc_hub)):
            hub_edge_features, hub_node_features = self.edge_node_fusion(
                node_attr=hub_x,
                edge_index=hub_adjacency_index,
                edge_attr=edge_features,
                dg=self.deep_gravity_dec_hub[1],
                gcn=self.gcn_dec_hub[1],
                cat_ln=self.concat_linear_dec_hub[1],
                adjacency_indices=adjacency_indices_lists[1],
                edge_mask=hub_edge_mask
            )
        hub_edge_features = self.edge_unpool(selected_edge_attr = hub_edge_features, original_num_edges = edge_features.shape[0], selected_edges=hub_edge_mask)

        # Process original layer
        x = torch.cat([node_features, weather_x], dim=-1)
        for i in range(len(self.gcn_enc)):
            edge_features, node_features = self.edge_node_fusion(
                node_attr=x,
                edge_index=adjacency_index,
                edge_attr=edge_features,
                dg=self.deep_gravity_dec[-1],
                gcn=self.gcn_dec[-1],
                cat_ln=self.concat_linear_dec[-1],
                adjacency_indices=adjacency_indices_lists[0],
                edge_mask=None
            )
        edge_features = torch.concat([top_edge_features, hub_edge_features, edge_features], dim=-1)

        return edge_features
    
    def edge_unpool(self, selected_edges, selected_edge_attr, original_num_edges):
        # edge features unpooling
        num_edge_features = selected_edge_attr.size(1)
        upsampled_edge_attr = torch.zeros(original_num_edges, num_edge_features).cuda().float()
        upsampled_edge_attr[selected_edges] = selected_edge_attr

        return upsampled_edge_attr

class WIMN(nn.Module):
    def __init__(self, node_input_dim, weather_input_dim, node_hidden_dims, edge_input_dim, edge_hidden_dims, dropout_rate):
        super(WIMN, self).__init__()

        self.previous_encoder = Encoder(
            node_input_dim=node_input_dim,
            node_hidden_dims=node_hidden_dims,
            edge_input_dim=edge_input_dim,
            edge_hidden_dims=edge_hidden_dims,
            dropout_rate=dropout_rate
        )

        self.next_decoder = Decoder(
            node_input_dim=node_hidden_dims[-1],
            weather_input_dim=weather_input_dim,
            node_hidden_dims=edge_hidden_dims,
            edge_input_dim=edge_hidden_dims[-1],
            edge_hidden_dims=edge_hidden_dims,
            dropout_rate=dropout_rate
        )
        concat_dims = edge_hidden_dims[-1]*3

        self.final_linear = nn.Linear(in_features=concat_dims, out_features=1)

    def forward(self, edge_index, interaction_index, edge_attr_unique, node_input_x, weather_inputs_x, adjacency_index, hub_indices_list, interaction_indices_lists, adjacency_indices_lists, hub_top_edge_mask):
        edge_hidden_z, node_hidden_z = self.previous_encoder(
            node_input_x=node_input_x,
            edge_index=edge_index,
            interaction_index=interaction_index,
            edge_attr=edge_attr_unique,
            interaction_indices_lists=interaction_indices_lists
        )

        decoder_edge_attr = self.next_decoder(
            edge_hidden_z=edge_hidden_z,
            node_hidden_z=node_hidden_z,
            weather_input_x=weather_inputs_x,
            adjacency_index=adjacency_index,
            hub_indices_list=hub_indices_list,
            adjacency_indices_lists=adjacency_indices_lists,
            hub_top_edge_mask=hub_top_edge_mask
        )

        next_prediction = self.final_linear(decoder_edge_attr).squeeze()
        return next_prediction
