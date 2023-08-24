import time

from torch import nn
import torch
import math
# from torch_geometric.nn import knn_graph
# from util_flow import cal_indices


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(inplace=True), attention=False,
                 add_bond_model=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention
        self.add_bond_model = add_bond_model

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            # nn.BatchNorm1d(hidden_nf),      # bn may incur problems when bs=1
            nn.LayerNorm(hidden_nf, elementwise_affine=True),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        if add_bond_model:
            self.bond_mlp = nn.Sequential(
                nn.Linear(input_edge + edges_in_d//2 + input_nf, hidden_nf),
                nn.LayerNorm(hidden_nf, elementwise_affine=True),
                act_fn,
                nn.Linear(hidden_nf, hidden_nf),
                act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            # nn.BatchNorm1d(hidden_nf),
            nn.LayerNorm(hidden_nf, elementwise_affine=True),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())
            if add_bond_model:
                self.bond_att_mlp = nn.Sequential(
                    nn.Linear(hidden_nf, 1),
                    nn.Sigmoid())

    def edge_model(self, source, target, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, edge_attr], dim=1)
        mij = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(mij)
            out = mij * att_val
        else:
            out = mij

        if edge_mask is not None:
            out = out * edge_mask
        return out, mij

    def bond_model(self, source, target, bond_attr):
        if bond_attr is None:  # Unused.
            out = torch.cat([source, target], dim=1)
        else:
            out = torch.cat([source, target, bond_attr], dim=1)
        bij = self.bond_mlp(out)

        if self.attention:
            bond_att_val = self.bond_att_mlp(bij)
            out = bij * bond_att_val
        else:
            out = bij

        return out, bij

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = x + self.node_mlp(agg)
        return out, agg

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None,
                bond_index=None, bond_attr=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        if bond_index is not None:
            assert self.add_bond_model is True, "add_bond_model should be True"
            bond_row, bond_col = bond_index
            bond_attr, bij = self.bond_model(h[bond_row], h[bond_col], bond_attr)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij, bond_attr


class EquivariantUpdate(nn.Module):
    def __init__(self, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=1, act_fn=nn.SiLU(), tanh=False, coords_range=10.0):
        super(EquivariantUpdate, self).__init__()
        self.tanh = tanh
        self.coords_range = coords_range
        input_edge = hidden_nf * 2 + edges_in_d
        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            # nn.BatchNorm1d(hidden_nf),  # nn.BatchNorm1d
            nn.LayerNorm(hidden_nf, elementwise_affine=True),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            nn.LayerNorm(hidden_nf, elementwise_affine=True),
            act_fn,
            layer)
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

    def coord_model(self, h, coord, edge_index, coord_diff, edge_attr, edge_mask=None):
        row, col = edge_index
        input_tensor = torch.cat([h[row], h[col], edge_attr], dim=1)
        if self.tanh:
            trans = coord_diff * torch.tanh(self.coord_mlp(input_tensor)) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(input_tensor)
        if edge_mask is not None:
            trans = trans * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        coord = coord + agg
        return coord

    def coord_model_bond(self, h, coord, bond_edge_index, bond_coord_diff, bond_attr, bond_edge_mask=None):
        row, col = bond_edge_index
        input_tensor = torch.cat([h[row], h[col], bond_attr], dim=1)
        if self.tanh:
            trans = bond_coord_diff * torch.tanh(self.coord_mlp_bond(input_tensor)) * self.coords_range
        else:
            trans = bond_coord_diff * self.coord_mlp_bond(input_tensor)
        if bond_edge_mask is not None:
            trans = trans * bond_edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0),
                                   normalization_factor=self.normalization_factor,
                                   aggregation_method=self.aggregation_method)
        # coord = coord + agg
        return agg

    def forward(self, h, coord, edge_index, coord_diff, edge_attr=None, node_mask=None, edge_mask=None,
                bond_edge_index=None, bond_coor_diff=None, bond_attr=None):
        coord = self.coord_model(h, coord, edge_index, coord_diff, edge_attr, edge_mask)
        if node_mask is not None:
            coord = coord * node_mask
        return coord


class EquivariantBlock(nn.Module):
    def __init__(self, hidden_nf, edge_feat_nf=2, device='cpu', act_fn=nn.SiLU(), n_layers=2, attention=True,
                 norm_diff=True, tanh=False, coords_range=15, norm_constant=1, sin_embedding=None,
                 normalization_factor=100, aggregation_method='sum', add_bond_model=False):
        super(EquivariantBlock, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range)
        self.norm_diff = norm_diff
        self.norm_constant = norm_constant
        self.sin_embedding = sin_embedding
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method

        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=edge_feat_nf,
                                              act_fn=act_fn, attention=attention,
                                              normalization_factor=self.normalization_factor,
                                              aggregation_method=self.aggregation_method,
                                              add_bond_model=add_bond_model))
        self.add_module("gcl_equiv", EquivariantUpdate(hidden_nf, edges_in_d=edge_feat_nf, act_fn=nn.SiLU(), tanh=tanh,
                                                       coords_range=self.coords_range_layer,
                                                       normalization_factor=self.normalization_factor,
                                                       aggregation_method=self.aggregation_method)
                                                       )
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None, edge_attr=None,
                bond_index=None, bond_attr=None):
        # Edit Emiel: Remove velocity as input
        distances, coord_diff = coord2diff(x, edge_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
        edge_attr = torch.cat([distances, edge_attr], dim=1)
        if bond_index is not None:
            distances_bond, bond_coord_diff = coord2diff(x, bond_index, self.norm_constant)
            if self.sin_embedding is not None:
                distances_bond = self.sin_embedding(distances_bond)
        else:
            bond_coord_diff = None
        for i in range(0, self.n_layers):
            h, _, bond_attr = self._modules["gcl_%d" % i](h, edge_index, edge_attr=edge_attr,
                                                          node_mask=node_mask, edge_mask=edge_mask,
                                                          bond_index=bond_index,
                                                          bond_attr=torch.cat([distances_bond, bond_attr], dim=1) if bond_index is not None else None)
        x = self._modules["gcl_equiv"](h, x, edge_index, coord_diff, edge_attr, node_mask, edge_mask,
                                       bond_edge_index=bond_index, bond_coor_diff=bond_coord_diff,
                                       bond_attr=torch.cat([distances_bond, bond_attr], dim=1) if bond_index is not None else None)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x, bond_attr


class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
                 norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
                 sin_embedding=False, normalization_factor=100, aggregation_method='sum',
                 add_bond_model=False):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range/n_layers)
        self.norm_diff = norm_diff
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.norm_constant = norm_constant

        if sin_embedding:
            self.sin_embedding = SinusoidsEmbeddingNew()
            edge_feat_nf = self.sin_embedding.dim * 2
        else:
            self.sin_embedding = None
            edge_feat_nf = 2

        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
                                                               act_fn=act_fn, n_layers=inv_sublayers,
                                                               attention=attention, norm_diff=norm_diff, tanh=tanh,
                                                               coords_range=coords_range, norm_constant=norm_constant,
                                                               sin_embedding=self.sin_embedding,
                                                               normalization_factor=self.normalization_factor,
                                                               aggregation_method=self.aggregation_method,
                                                               add_bond_model=add_bond_model)
                            )
        self.to(self.device)

    def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
                bond_index=None, bond_attr=None):
        # num_ligand_atoms denotes all atoms of possibly multiple ligands for a given pocket
        # Edit Emiel: Remove velocity as input
        distances, _ = coord2diff(x, edge_index)
        # distances_bond, bond_coord_diff = coord2diff(x, bond_index, self.norm_constant)
        if self.sin_embedding is not None:
            distances = self.sin_embedding(distances)
            # distances_bond = self.sin_embedding(distances_bond)
        h = self.embedding(h)
        # edge_attr = edge_attr0
        for i in range(0, self.n_layers):
            h, x, bond_attr = self._modules["e_block_%d" % i](h, x, edge_index.to(h.device), node_mask=node_mask,
                                                              edge_mask=edge_mask, edge_attr=distances,
                                                              bond_index=bond_index, bond_attr=bond_attr)
        # Important, the bias of the last linear might be non-zero
        h = self.embedding_out(h)
        if node_mask is not None:
            h = h * node_mask
        if bond_index is not None:
            # bond_attr = self.embedding_out(bond_attr)
            return h, x, bond_attr
        return h, x

# class EGNN(nn.Module):
#     def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=3, attention=False,
#                  norm_diff=True, out_node_nf=None, tanh=False, coords_range=15, norm_constant=1, inv_sublayers=2,
#                  sin_embedding=False, normalization_factor=100, aggregation_method='sum',
#                  add_bond_model=False, pos_bond=False, use_latest_h=True, dynamic_edge=False):
#         super(EGNN, self).__init__()
#         if out_node_nf is None:
#             out_node_nf = in_node_nf
#         self.hidden_nf = hidden_nf
#         self.device = device
#         self.n_layers = n_layers
#         self.coords_range_layer = float(coords_range/n_layers)
#         self.norm_diff = norm_diff
#         self.normalization_factor = normalization_factor
#         self.aggregation_method = aggregation_method
#         self.dynamic_edge = dynamic_edge
#         self.norm_constant = norm_constant
#
#         if sin_embedding:
#             self.sin_embedding = SinusoidsEmbeddingNew()
#             edge_feat_nf = self.sin_embedding.dim * 2
#             self.bond_attr2edge_attr = nn.Linear(hidden_nf, self.sin_embedding.dim)
#             self.edge_attr_dist_linear = nn.Linear(self.sin_embedding.dim * 2, self.sin_embedding.dim)
#             self.bond_attr_dist_linear = nn.Linear(self.sin_embedding.dim + hidden_nf, hidden_nf)
#         else:
#             self.sin_embedding = None
#             edge_feat_nf = 2
#
#         self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
#         self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
#         for i in range(0, n_layers):
#             self.add_module("e_block_%d" % i, EquivariantBlock(hidden_nf, edge_feat_nf=edge_feat_nf, device=device,
#                                                                act_fn=act_fn, n_layers=inv_sublayers,
#                                                                attention=attention, norm_diff=norm_diff, tanh=tanh,
#                                                                coords_range=coords_range, norm_constant=norm_constant,
#                                                                sin_embedding=self.sin_embedding,
#                                                                normalization_factor=self.normalization_factor,
#                                                                aggregation_method=self.aggregation_method,
#                                                                add_bond_model=add_bond_model,
#                                                                pos_bond=pos_bond, use_latest_h=use_latest_h)
#                             )
#             # self.add_module("e_block_linear_%d" % i, nn.Linear(self.hidden_nf + time_dim, self.hidden_nf))
#             # if add_bond_model:
#             #     self.add_module("e_block_bond_linear_%d" % i, nn.Linear(self.hidden_nf + time_dim, self.hidden_nf))
#         self.to(self.device)
#
#     def forward(self, h, x, edge_index, node_mask=None, edge_mask=None,
#                 bond_index=None, bond_attr=None, t=None, n_knn=None,
#                 batch_protein=None, batch_ligand=None, num_ligand_atoms=None, num_atoms_per_ligand=None):
#         # num_ligand_atoms denotes all atoms of possibly multiple ligands for a given pocket
#         # Edit Emiel: Remove velocity as input
#         distances, _ = coord2diff(x, edge_index)
#         distances_bond, bond_coord_diff = coord2diff(x, bond_index, self.norm_constant)
#         if self.sin_embedding is not None:
#             edge_attr0 = torch.ones((edge_index.shape[1], self.sin_embedding.dim), device=x.device) * 0.1
#             # torch.cuda.synchronize()
#             if num_atoms_per_ligand < 60:
#                 # t1 = time.time()
#                 res = cal_indices(edge_index.type(torch.int16), bond_index.type(torch.int16))  # .cpu()
#                 edge_attr_indices, bond_attr_indices = res[:, 0], res[:, 1]
#                 edge_attr0[edge_attr_indices] = self.bond_attr2edge_attr(bond_attr)[bond_attr_indices]
#                 # torch.cuda.synchronize()
#                 # print("faster time: %.3f %d" % (time.time() - t1, num_ligand_atoms/8))
#             else:
#                 t1 = time.time()
#                 bond_index_, edge_index_ = bond_index.cpu(), edge_index.cpu()
#                 bond_edge_dict = {(row.item(), col.item()): idx for idx, (row, col) in
#                                   enumerate(zip(bond_index_[0], bond_index_[1]))}
#                 bond_attr_indices, edge_attr_indices = [], []
#                 for edge_order, (row, col) in enumerate(zip(edge_index_[0], edge_index_[1])):
#                     if col < num_ligand_atoms:
#                         bond_attr_indices.append(bond_edge_dict[(row.item(), col.item())])
#                         edge_attr_indices.append(edge_order)
#                 edge_attr0[edge_attr_indices] = self.bond_attr2edge_attr(bond_attr)[bond_attr_indices]  # .cpu()
#                 # torch.cuda.synchronize()
#                 print("----------------edge attr time: %.3f %d-------------" % (time.time() - t1, num_ligand_atoms))
#             distances = self.sin_embedding(distances)
#             distances_bond = self.sin_embedding(distances_bond)
#         h = self.embedding(h)
#         edge_attr = edge_attr0
#         for i in range(0, self.n_layers):
#             h, x, bond_attr = self._modules["e_block_%d" % i](h, x, edge_index.to(h.device), node_mask=node_mask,
#                                                    edge_mask=edge_mask, edge_attr=edge_attr.to(x.device),
#                                                    bond_index=bond_index, bond_attr=bond_attr)
#             if i < self.n_layers - 1:
#                 edge_attr = self.edge_attr_dist_linear(torch.cat([edge_attr0, distances], dim=1))
#                 bond_attr = self.bond_attr_dist_linear(torch.cat([bond_attr, distances_bond], dim=1))
#         # Important, the bias of the last linear might be non-zero
#         h = self.embedding_out(h)
#         if node_mask is not None:
#             h = h * node_mask
#         if bond_index is not None:
#             # bond_attr = self.embedding_out(bond_attr)
#             return h, x, bond_attr
#         return h, x


class GNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, aggregation_method='sum', device='cpu',
                 act_fn=nn.SiLU(), n_layers=4, attention=False,
                 normalization_factor=1, out_node_nf=None):
        super(GNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        ### Encoder
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(
                self.hidden_nf, self.hidden_nf, self.hidden_nf,
                normalization_factor=normalization_factor,
                aggregation_method=aggregation_method,
                edges_in_d=in_edge_nf, act_fn=act_fn,
                attention=attention))
        self.to(self.device)

    def forward(self, h, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h


class SinusoidsEmbeddingNew(nn.Module):
    def __init__(self, max_res=15., min_res=15. / 2000., div_factor=4):
        super().__init__()
        self.n_frequencies = int(math.log(max_res / min_res, div_factor)) + 1
        self.frequencies = 2 * math.pi * div_factor ** torch.arange(self.n_frequencies)/max_res
        self.dim = len(self.frequencies) * 2

    def forward(self, x):
        x = torch.sqrt(x + 1e-8)
        emb = x * self.frequencies[None, :].to(x.device)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb # emb.detach()


def coord2diff(x, edge_index, norm_constant=1):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    norm = torch.sqrt(radial + 1e-8)
    coord_diff = coord_diff/(norm + norm_constant)
    return radial, coord_diff


def unsorted_segment_sum(data, segment_ids, num_segments, normalization_factor, aggregation_method: str):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
        Normalization: 'sum' or 'mean'.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    if aggregation_method == 'sum':
        result = result / normalization_factor

    if aggregation_method == 'mean':
        norm = data.new_zeros(result.shape)
        norm.scatter_add_(0, segment_ids, data.new_ones(data.shape))
        norm[norm == 0] = 1
        result = result / norm
    return result


