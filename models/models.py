import torch
from sys import platform
import torch.nn as nn
from torch_geometric.nn import knn_graph
import numpy as np
from utils.util_flow import subtract_mean, get_all_edges_faster
from torch_geometric.nn.pool import global_mean_pool
from .egnn_bond import EGNN


class Model(nn.Module):
    def __init__(self, num_protein_element=5, num_amino_acid=20, dim_protein_element=16, dim_amino_acid=16,
                 dim_is_backbone=8, num_ligand_element=7, num_bond_types=5, depth=5, hidden_dim=256, n_steps=1000,
                 use_mlp=True, gcl_layers=1, feat_time_dim=8, optimize_embedding=False, agg_norm_factor=5.0,
                 use_latest_h=True, distance_sin_emb=True, egnn_attn=True, n_knn=48, device='cpu',
                 add_bond_model=True, opt_time_emb=False, ):
        super().__init__()
        self.depth = depth
        self.dim_protein_element = dim_protein_element
        self.dim_amino_acid = dim_amino_acid
        self.dim_is_backbone = dim_is_backbone  # bool: True or False
        self.num_ligand_element = num_ligand_element
        self.use_mlp = use_mlp
        self.feat_time_dim = feat_time_dim
        self.protein_feat_dim = dim_protein_element + dim_amino_acid + dim_is_backbone
        self.device = device
        self.ligand_emb_dim = self.protein_feat_dim - feat_time_dim
        self.optimize_embedding = optimize_embedding  # if true, all embeddings will be optimized
        self.hidden_dim = hidden_dim  # EGNN hidden dim
        self.distance_sin_emb = distance_sin_emb
        self.n_knn = n_knn
        self.add_bond_model = add_bond_model
        self.num_timesteps = n_steps
        # construct embeddings
        self.time_embedding_feat = nn.Embedding(self.num_timesteps, self.feat_time_dim).requires_grad_(opt_time_emb)
        self.ligand_element_embedding = nn.Embedding(num_ligand_element, self.ligand_emb_dim).requires_grad_(
            optimize_embedding)
        self.protein_ele_embedding = nn.Embedding(num_protein_element, dim_protein_element).requires_grad_(
            optimize_embedding)
        self.protein_amino_acid_embedding = nn.Embedding(num_amino_acid, dim_amino_acid).requires_grad_(
            optimize_embedding)
        self.protein_is_backbone_embedding = nn.Embedding(2, dim_is_backbone).requires_grad_(optimize_embedding)
        self.bond_type_embedding = nn.Embedding(num_bond_types, self.ligand_emb_dim).requires_grad_(optimize_embedding)
        self.egnn = EGNN(self.protein_feat_dim, None,
                         hidden_nf=hidden_dim, device=device, n_layers=depth, out_node_nf=self.ligand_emb_dim,
                         sin_embedding=distance_sin_emb, attention=egnn_attn, add_bond_model=add_bond_model,
                         normalization_factor=agg_norm_factor, inv_sublayers=gcl_layers, use_latest_h=use_latest_h)
        if add_bond_model:
            self.bond_emb_input_linear = nn.Linear(self.protein_feat_dim, hidden_dim)
            # self.bond_emb_output_linear = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, self.ligand_emb_dim))
            self.bond_emb_output_linear = nn.Linear(hidden_dim, self.ligand_emb_dim)

        if use_mlp:
            self.pos_linear = nn.Linear(1, 1, bias=False)
            self.pos_linear.weight.data = torch.ones((1, 1), device=device)
            # self.pos_linear.bias.data = torch.zeros((1, 1), device=device)  # comment out this line to preserve equivariance

    def infer_indices(self, edges, num_atoms_per_ligand):
        '''
        Given an index (i,j), its position in bond_index is i*(N-1) + j - int(i<j), where N is the number of atoms in
        one ligand. We can generalize it to multiple atoms with i*(N-1) + (j-int(i/N)*N) - int(i<j). The fundamental
        formula is from Xingang Peng.
        '''
        term1 = edges[0] * (num_atoms_per_ligand - 1)
        term2 = edges[1] - (edges[0] / num_atoms_per_ligand).long() * num_atoms_per_ligand
        term3 = (edges[0] < edges[1]).long()
        return term1 + term2 - term3

    # https://discuss.pytorch.org/t/how-to-convert-adjacency-matrix-to-edge-index-format/145239
    # https://discuss.pytorch.org/t/fill-diagonal-of-matrix-with-zero/35083/6
    # @staticmethod
    def get_ligand_bond_types_all(self, num_ligand_atoms, ligand_bond_indices=None, ligand_bond_types=None,
                                  device='cuda'):
        adj_matrix = torch.ones((num_ligand_atoms, num_ligand_atoms))
        adj_matrix.fill_diagonal_(0)
        bond_edges = adj_matrix.nonzero().t()  # a tricky operation
        if ligand_bond_types is not None:
            ligand_bond_types_all = torch.zeros((bond_edges.size(1),), dtype=torch.long, device=device)
            edge_attr_indices_ = self.infer_indices(ligand_bond_indices, num_atoms_per_ligand=num_ligand_atoms).to(
                self.device)
            ligand_bond_types_all[edge_attr_indices_] = ligand_bond_types.long().to(device)
            return ligand_bond_types_all, bond_edges
        else:
            return bond_edges

    def get_ligand_features(self, ligand_elements):
        ligand_elements_embedding = self.ligand_element_embedding(
            ligand_elements.to(self.device))  # * self.emb_scale_factor
        return ligand_elements_embedding

    def get_bond_type_features(self, bond_types):
        bond_type_features = self.bond_type_embedding(bond_types.to(self.device))  # * self.emb_scale_factor
        return bond_type_features

    def get_X0(self, num_ligand_atoms, num_edges=None, batchsize=1, batch_ligands=None):
        X0_pos = torch.randn(size=(batchsize * num_ligand_atoms, 3), device=self.device)
        X0_pos = subtract_mean(X0_pos, verbose=False, batch=batch_ligands)  # zero gravity
        X0_element_embedding = torch.randn(size=(batchsize * num_ligand_atoms, self.ligand_emb_dim),
                                           device=self.device)  # * self.emb_scale_factor
        if num_edges is not None:
            X0_bond_embedding = torch.randn(size=(num_edges, self.ligand_emb_dim), device=self.device)
            return X0_pos, X0_element_embedding, X0_bond_embedding
        return X0_pos, X0_element_embedding, None

    def forward(self, protein_positions, protein_ele, protein_amino_acid, protein_is_backbone, Xt_pos, Xt_features, t,
                batch_protein=None, batch_ligand=None, bond_features=None, bond_indices=None,
                num_proposals=1, num_atoms_per_ligand=None, n_steps_per_iter=1, t_bond=None, ):
        protein_ele_emb = self.protein_ele_embedding(protein_ele)
        protein_aa_emb = self.protein_amino_acid_embedding(protein_amino_acid)
        protein_is_backbone_emb = self.protein_is_backbone_embedding(protein_is_backbone.int())
        protein_features = torch.cat([protein_ele_emb, protein_aa_emb, protein_is_backbone_emb], dim=-1)
        feat_time_embedding = self.time_embedding_feat(t)
        feats = torch.cat([Xt_features, feat_time_embedding], dim=-1)
        feats = torch.cat([feats, protein_features], dim=0)
        coors = torch.cat([Xt_pos, protein_positions], dim=0)
        if n_steps_per_iter > 1 or num_proposals > 1:
            if platform != 'win32':
                edges = get_all_edges_faster(Xt_pos.cpu(), protein_positions.cpu(), k=self.n_knn,
                                             num_proposals=num_proposals, num_ligand_atoms=num_atoms_per_ligand,
                                             num_timesteps=n_steps_per_iter).to(self.device)
            else:
                edges = get_all_edges_faster(Xt_pos, protein_positions, k=self.n_knn,
                                             num_proposals=num_proposals, num_ligand_atoms=num_atoms_per_ligand,
                                             num_timesteps=n_steps_per_iter).to(self.device)
        else:
            if platform != 'win32':
                edges = knn_graph(coors.cpu(), k=self.n_knn, flow="target_to_source", num_workers=20, loop=False,
                                  batch=torch.cat([batch_ligand, batch_protein]) if batch_protein is not None else None)
            else:
                edges = knn_graph(coors, k=self.n_knn, flow="target_to_source", num_workers=20, loop=False,
                                  batch=torch.cat([batch_ligand, batch_protein]) if batch_protein is not None else None)
            edges = edges[:, :Xt_pos.shape[0] * self.n_knn].to(self.device)  # We only update ligand atoms.

        if bond_features is not None:
            feat_time_embedding = self.time_embedding_feat(t_bond)
            bond_features = torch.cat([bond_features, feat_time_embedding], dim=-1)
            bond_features = self.bond_emb_input_linear(bond_features)
            feats, coors, bond_features = self.egnn(feats, coors, edges, bond_index=bond_indices,
                                                    bond_attr=bond_features)
            bond_features = self.bond_emb_output_linear(bond_features)
        else:
            feats, coors = self.egnn(feats, coors, edges, bond_index=None, bond_attr=None)
        coors = coors[:Xt_pos.shape[0]]  # get the ligand position values
        if self.use_mlp:
            coors = self.pos_linear(coors.unsqueeze(-1)).squeeze(-1)  # this is a linear vector neuron operation.

        return coors, feats[:Xt_pos.shape[0]], bond_features  # get the predicted ligand element embeddings


class Model_CoM_free(nn.Module):
    def __init__(self, num_protein_element=5, num_amino_acid=20, dim_protein_element=16, dim_amino_acid=16,
                 dim_is_backbone=8, num_ligand_element=7, num_bond_types=5, depth=5, hidden_dim=256, n_steps=1000,
                 use_mlp=True, gcl_layers=1, feat_time_dim=8, optimize_embedding=False, agg_norm_factor=5.0,
                 distance_sin_emb=True, egnn_attn=True, n_knn=48, device='cpu', add_bond_model=True,
                 opt_time_emb=False):
        super().__init__()
        self.depth = depth
        self.dim_protein_element = dim_protein_element
        self.dim_amino_acid = dim_amino_acid
        self.dim_is_backbone = dim_is_backbone  # bool: True or False
        self.num_ligand_element = num_ligand_element
        self.use_mlp = use_mlp
        self.feat_time_dim = feat_time_dim
        self.protein_feat_dim = dim_protein_element + dim_amino_acid + dim_is_backbone
        self.device = device
        self.ligand_emb_dim = self.protein_feat_dim - feat_time_dim
        self.optimize_embedding = optimize_embedding  # if true, all embeddings will be optimized
        self.hidden_dim = hidden_dim  # EGNN hidden dim
        self.distance_sin_emb = distance_sin_emb
        self.n_knn = n_knn
        self.add_bond_model = add_bond_model
        self.num_timesteps = n_steps
        # construct embeddings
        self.time_embedding_feat = nn.Embedding(self.num_timesteps, self.feat_time_dim).requires_grad_(opt_time_emb)
        self.ligand_element_embedding = nn.Embedding(num_ligand_element, self.ligand_emb_dim).requires_grad_(
            optimize_embedding)
        self.protein_ele_embedding = nn.Embedding(num_protein_element, dim_protein_element).requires_grad_(
            optimize_embedding)
        self.protein_amino_acid_embedding = nn.Embedding(num_amino_acid, dim_amino_acid).requires_grad_(
            optimize_embedding)
        self.protein_is_backbone_embedding = nn.Embedding(2, dim_is_backbone).requires_grad_(optimize_embedding)
        self.bond_type_embedding = nn.Embedding(num_bond_types, self.ligand_emb_dim).requires_grad_(optimize_embedding)
        self.egnn = EGNN(self.protein_feat_dim, None,
                         hidden_nf=hidden_dim, device=device, n_layers=depth, out_node_nf=self.ligand_emb_dim,
                         sin_embedding=distance_sin_emb, attention=egnn_attn, add_bond_model=add_bond_model,
                         normalization_factor=agg_norm_factor, inv_sublayers=gcl_layers)

        self.bond_emb_input_linear = nn.Linear(self.protein_feat_dim, hidden_dim)
        self.bond_emb_output_linear = nn.Sequential(nn.SiLU(),
                                                    nn.Linear(hidden_dim, self.ligand_emb_dim))

        if use_mlp:
            self.pos_linear = nn.Linear(1, 1)
            self.pos_linear.weight.data = torch.ones((1, 1), device=device)
            self.pos_linear.bias.data = torch.zeros((1, 1), device=device)

    def infer_indices(self, edges, num_atoms_per_ligand):
        '''
        Given an index (i,j), its position in bond_index is i*(N-1) + j - int(i<j), where N is the number of atoms in
        one ligand. We can generalize it to multiple atoms with i*(N-1) + (j-int(i/N)*N) - int(i<j). The fundamental
        formula is from Xingang Peng.
        '''
        term1 = edges[0] * (num_atoms_per_ligand - 1)
        term2 = edges[1] - (edges[0] / num_atoms_per_ligand).long() * num_atoms_per_ligand
        term3 = (edges[0] < edges[1]).long()
        return term1 + term2 - term3

    # https://discuss.pytorch.org/t/how-to-convert-adjacency-matrix-to-edge-index-format/145239
    # https://discuss.pytorch.org/t/fill-diagonal-of-matrix-with-zero/35083/6
    # @staticmethod
    def get_ligand_bond_types_all(self, num_ligand_atoms, ligand_bond_indices=None, ligand_bond_types=None,
                                  device='cuda'):
        adj_matrix = torch.ones((num_ligand_atoms, num_ligand_atoms))
        adj_matrix.fill_diagonal_(0)
        bond_edges = adj_matrix.nonzero().t()  # a tricky operation
        if ligand_bond_types is not None:
            ligand_bond_types_all = torch.zeros((bond_edges.size(1),), dtype=torch.long, device=device)
            edge_attr_indices_ = self.infer_indices(ligand_bond_indices, num_atoms_per_ligand=num_ligand_atoms).to(
                self.device)
            ligand_bond_types_all[edge_attr_indices_] = ligand_bond_types.to(device).long()
            return ligand_bond_types_all, bond_edges
        else:
            return bond_edges

    def get_ligand_features(self, ligand_elements):
        ligand_elements_embedding = self.ligand_element_embedding(
            ligand_elements.to(self.device))  # * self.emb_scale_factor
        return ligand_elements_embedding

    def get_bond_type_features(self, bond_types):
        bond_type_features = self.bond_type_embedding(bond_types.to(self.device))  # * self.emb_scale_factor
        return bond_type_features

    def get_X0(self, num_ligand_atoms, num_edges=None, batchsize=1, batch_ligands=None):
        X0_pos = torch.randn(size=(batchsize * num_ligand_atoms, 3), device=self.device)
        X0_pos = subtract_mean(X0_pos, verbose=False, batch=batch_ligands)  # zero gravity
        X0_element_embedding = torch.randn(size=(batchsize * num_ligand_atoms, self.ligand_emb_dim),
                                           device=self.device)  # * self.emb_scale_factor
        if num_edges is not None:
            X0_bond_embedding = torch.randn(size=(num_edges, self.ligand_emb_dim), device=self.device)
            return X0_pos, X0_element_embedding, X0_bond_embedding
        return X0_pos, X0_element_embedding, None

    def forward(self, protein_positions, protein_ele, protein_amino_acid, protein_is_backbone, Xt_pos, Xt_features, t,
                batch_protein=None, batch_ligand=None, bond_features=None, bond_indices=None,
                num_proposals=1, num_atoms_per_ligand=None, n_steps_per_iter=1, t_bond=None, ):
        protein_ele_emb = self.protein_ele_embedding(protein_ele)
        protein_aa_emb = self.protein_amino_acid_embedding(protein_amino_acid)
        protein_is_backbone_emb = self.protein_is_backbone_embedding(protein_is_backbone.int())
        protein_features = torch.cat([protein_ele_emb, protein_aa_emb, protein_is_backbone_emb], dim=-1)
        feat_time_embedding = self.time_embedding_feat(t)
        feats = torch.cat([Xt_features, feat_time_embedding], dim=-1)
        feats = torch.cat([feats, protein_features], dim=0)
        coors = torch.cat([Xt_pos, protein_positions], dim=0)
        if n_steps_per_iter > 1 or num_proposals > 1:
            if platform != 'win32':
                edges = get_all_edges_faster(Xt_pos.cpu(), protein_positions.cpu(), k=self.n_knn,
                                             num_proposals=num_proposals, num_ligand_atoms=num_atoms_per_ligand,
                                             num_timesteps=n_steps_per_iter).to(self.device)
            else:
                edges = get_all_edges_faster(Xt_pos, protein_positions, k=self.n_knn,
                                             num_proposals=num_proposals, num_ligand_atoms=num_atoms_per_ligand,
                                             num_timesteps=n_steps_per_iter).to(self.device)
        else:
            if platform != 'win32':
                edges = knn_graph(coors.cpu(), k=self.n_knn, flow="target_to_source", num_workers=20, loop=False,
                                  batch=torch.cat([batch_ligand, batch_protein]) if batch_protein is not None else None)
            else:
                edges = knn_graph(coors, k=self.n_knn, flow="target_to_source", num_workers=20, loop=False,
                                  batch=torch.cat([batch_ligand, batch_protein]) if batch_protein is not None else None)
            edges = edges[:, :Xt_pos.shape[0] * self.n_knn].to(self.device)  # We only update ligand atoms.

        if bond_features is not None:
            feat_time_embedding = self.time_embedding_feat(t_bond)
            bond_features = torch.cat([bond_features, feat_time_embedding], dim=-1)
            bond_features = self.bond_emb_input_linear(bond_features)
            feats, coors, bond_features = self.egnn(feats, coors, edges, bond_index=bond_indices,
                                                    bond_attr=bond_features)
            bond_features = self.bond_emb_output_linear(bond_features)
        else:
            feats, coors = self.egnn(feats, coors, edges, bond_index=None, bond_attr=None)
        coors = coors[:Xt_pos.shape[0]]  # get the ligand position values
        if self.use_mlp:
            coors = self.pos_linear(coors.unsqueeze(-1)).squeeze(-1)  # this is a linear vector neuron operation.

        return coors, feats[:Xt_pos.shape[0]], bond_features  # get the predicted ligand element embeddings


def coord2diff(x, edge_index):
    row, col = edge_index
    coord_diff = x[row] - x[col]
    radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)
    return radial

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


class GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, normalization_factor, aggregation_method,
                 edges_in_d=0, nodes_att_dim=0, act_fn=nn.SiLU(inplace=True), attention=False):
        super(GCL, self).__init__()
        input_edge = input_nf * 2
        self.normalization_factor = normalization_factor
        self.aggregation_method = aggregation_method
        self.attention = attention

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edges_in_d, hidden_nf),
            # nn.BatchNorm1d(hidden_nf),      # bn may incur problems when bs=1
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

    def forward(self, h, edge_index, edge_attr=None, node_attr=None, node_mask=None, edge_mask=None):
        row, col = edge_index
        edge_feat, mij = self.edge_model(h[row], h[col], edge_attr, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        if node_mask is not None:
            h = h * node_mask
        return h, mij


class Classifier(nn.Module):
    def __init__(self, in_node_feat, hidden_node_feat, out_node_feat, time_emb_feat, depth, num_timesteps, device,
                 opt_time_emb=False, n_knn=8, add_bond=False, num_gaussian=20, use_pocket=False, **kwargs):
        super(Classifier, self).__init__()
        self.device = device
        self.embedding_in = nn.Linear(in_node_feat + time_emb_feat, hidden_node_feat)
        self.embedding_out = nn.Linear(hidden_node_feat, out_node_feat)
        self.out = nn.Linear(out_node_feat, 1)
        self.time_embedding = nn.Embedding(num_timesteps, time_emb_feat).requires_grad_(opt_time_emb)
        self.n_knn = n_knn
        self.add_bond = add_bond
        self.edge_in_d = num_gaussian + in_node_feat + time_emb_feat if add_bond else num_gaussian + time_emb_feat
        layers = []
        self.use_pocket = use_pocket
        if use_pocket:
            num_protein_element, dim_protein_element = kwargs['num_protein_element'], kwargs['dim_protein_element']
            num_amino_acid, dim_amino_acid = kwargs['num_amino_acid'], kwargs['dim_amino_acid']
            dim_is_backbone = kwargs['dim_is_backbone']
            # dim_unspecified_bond_embedding = dim_protein_element + dim_amino_acid + dim_is_backbone
            self.protein_ele_embedding = nn.Embedding(num_protein_element, dim_protein_element).requires_grad_(False)
            self.protein_amino_acid_embedding = nn.Embedding(num_amino_acid, dim_amino_acid).requires_grad_(False)
            self.protein_is_backbone_embedding = nn.Embedding(2, dim_is_backbone).requires_grad_(False)
            self.unspecified_bond_embedding = nn.Embedding(2, in_node_feat).requires_grad_(False)
        for i in range(depth):
            layers.append(GCL(input_nf=hidden_node_feat,
                              output_nf=hidden_node_feat,
                              hidden_nf=hidden_node_feat,
                              normalization_factor=5,
                              aggregation_method='sum',
                              edges_in_d=self.edge_in_d,
                              attention=True)
                          )
        self.gcl_layers = nn.ModuleList(layers)
        self.gaussian_smearing = GaussianSmearing(num_gaussians=num_gaussian)
        self.to(device)

    def forward(self, x, h, t, num_atoms_per_ligand, bond_features=None, t_bond=None, batch_ligand=None, **kwargs):
        time_emb = self.time_embedding(t.to(self.device))
        h = torch.cat([h, time_emb], dim=-1)
        num_ligand_atoms = x.shape[0]
        if self.use_pocket:
            protein_positions = kwargs['protein_positions']
            x = torch.cat([x, protein_positions], dim=0)
            protein_ele_emb = self.protein_ele_embedding(kwargs['protein_ele'])
            protein_aa_emb = self.protein_amino_acid_embedding(kwargs['protein_amino_acid'])
            protein_is_backbone_emb = self.protein_is_backbone_embedding(kwargs['protein_is_backbone'].int())
            protein_features = torch.cat([protein_ele_emb, protein_aa_emb, protein_is_backbone_emb], dim=-1)
            h = torch.cat([h, protein_features], dim=0)
            edges = get_all_edges_faster(x, protein_positions, k=self.n_knn, num_proposals=kwargs['num_proposals'],
                                         num_ligand_atoms=num_atoms_per_ligand,
                                         num_timesteps=kwargs['n_steps_per_iter']).to(self.device)
        else:
            edges = knn_graph(x.cpu(), k=self.n_knn, flow="target_to_source", num_workers=18, loop=False,
                              batch=batch_ligand.cpu() if batch_ligand is not None else None).to(self.device)
        if self.use_pocket:
            edges = edges[:, :(num_ligand_atoms * self.n_knn)]
        distances = coord2diff(x, edge_index=edges)
        selected_distance_id = distances < 7
        distances = distances[selected_distance_id]  # check the dimension to guarantee!!!
        edge_mask = selected_distance_id.to(self.device)
        edges = edges[:, edge_mask.squeeze()]
        if self.use_pocket:
            edge_id_bond = edges[1] < num_ligand_atoms  # boolean, to indicate edges between ligand atoms
            bond_edges = edges[:, edge_id_bond]  # exclude the edge between ligand and pocket
            selected_bond_indices = bond_edges[0] * (num_atoms_per_ligand - 1) + bond_edges[1] - \
                                    (bond_edges[0] / num_atoms_per_ligand).long() * num_atoms_per_ligand - \
                                    (bond_edges[0] < bond_edges[1]).long()  # This idea is from Xingang Peng
        else:
            selected_bond_indices = edges[0] * (num_atoms_per_ligand - 1) + edges[1] - \
                                    (edges[0] / num_atoms_per_ligand).long() * num_atoms_per_ligand - \
                                    (edges[0] < edges[1]).long()  # This idea is from Xingang Peng

        distances = self.gaussian_smearing(torch.sqrt(distances))
        if bond_features is not None and self.add_bond:
            # assert bond_indices is not None, "bond_indices should be specified if bond_features is not None."
            selected_bond_attr = bond_features[selected_bond_indices.to(self.device)]
            if self.use_pocket:
                edge_feat_with_pocket = (
                        self.unspecified_bond_embedding(torch.tensor(0).to(self.device)) * 0.25).repeat(
                    edges.shape[1], 1)
                edge_feat_with_pocket[edge_id_bond] = selected_bond_attr
                selected_bond_attr = edge_feat_with_pocket
            edge_time_emb = self.time_embedding(t_bond[0]).repeat(edges.shape[1], 1)  # only support a single time step
            edge_attr = torch.cat([selected_bond_attr, edge_time_emb, distances], dim=-1)
        else:
            edge_time_emb = self.time_embedding(t_bond[selected_bond_indices.to(self.device)])
            edge_attr = torch.cat([edge_time_emb, distances], dim=-1)
        h = self.embedding_in(h)
        for i, gcl in enumerate(self.gcl_layers):
            h = gcl(h, edges, edge_attr=edge_attr, edge_mask=None)[0]
        h = self.embedding_out(h)
        h = global_mean_pool(h[:num_ligand_atoms], batch=batch_ligand.to(self.device))
        h = self.out(h).squeeze()
        return h


class GaussianSmearing(nn.Module):
    # from Xingang Peng's MolDiff code
    def __init__(self, start=0.0, stop=4.0, num_gaussians=20, type_='exp'):
        super().__init__()
        self.start = start
        self.stop = stop
        if type_ == 'exp':
            offset = torch.exp(torch.linspace(start=np.log(start + 1), end=np.log(stop + 1), steps=num_gaussians)) - 1
        elif type_ == 'linear':
            offset = torch.linspace(start=start, end=stop, steps=num_gaussians)
        else:
            raise NotImplementedError('type_ must be either exp or linear')
        diff = torch.diff(offset)
        diff = torch.cat([diff[:1], diff])
        coeff = -0.5 / (diff ** 2)
        self.register_buffer('coeff', coeff)
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = torch.clamp(dist, min=self.start, max=self.stop)
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))
