import torch
import torch.nn.functional as F
import numpy as np

from datasets.pl_data import ProteinLigandData
from utils import data as utils_data

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX_wo_h = {
    6: 0,
    7: 1,
    8: 2,
    9: 3,
    15: 4,
    16: 5,
    17: 6,
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h = {
    # (1, False): 0,
    (6, False): 0,
    (6, True): 1,
    (7, False): 2,
    (7, True): 3,
    (8, False): 4,
    (8, True): 5,
    (9, False): 6,
    (15, False): 7,
    (15, True): 8,
    (16, False): 9,
    (16, True): 10,
    (17, False): 11
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_ONLY_wo_h = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX_wo_h.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC_wo_h = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}

def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'basic_wo_h':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY_wo_h[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'add_aromatic_wo_h':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_wo_h[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'add_aromatic_wo_h':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_wo_h[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic' or mode == 'basic_wo_h':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    if mode == 'full_wo_h':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC_wo_h[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'basic_wo_h':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX_wo_h[int(atom_num)]
    elif mode == 'add_aromatic':
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
    elif mode == 'add_aromatic_wo_h':
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h[int(atom_num), bool(is_aromatic)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]


class FeaturizeProteinAtom(object):

    def __init__(self, without_Se=False):
        super().__init__()
        self.without_Se = without_Se
        if without_Se:    # The training data includes [1, 6, 7, 8, 16]
            self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16])  # H, C, N, O, S
        else:
            self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def feature_dim(self):
        if self.without_Se:
            return self.atomic_numbers.size(0) + self.max_num_aa + 2
        else:
            return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type.long(), num_classes=self.max_num_aa)
        if self.without_Se:
            is_backbone = F.one_hot(data.protein_is_backbone.long(), num_classes=2)
        else:
            is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x   # one-hot
        return data


ProteinElement2IndexDict = {1: 0, 6: 1, 7: 2, 8: 3, 16: 4}


class FeaturizeProteinELement(object):
    '''To get the index of protein element type ([1, 6, 7, 8, 16]) by Youming Zhao'''
    def __init__(self, ):
        super().__init__()

    @property
    def feature_dim(self):
        return len(ProteinElement2IndexDict)

    def __call__(self, data: ProteinLigandData):
        element_list = data.protein_element
        x = [ProteinElement2IndexDict[int(ele)] for ele in element_list]
        x = torch.tensor(x)
        data.protein_atom_element_index = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic_wo_h'):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full', 'basic_wo_h', 'add_aromatic_wo_h', 'full_wo_h']
        self.mode = mode

    @property
    def feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        elif self.mode == 'basic_wo_h':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX_wo_h)
        elif self.mode == 'add_aromatic_wo_h':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX_wo_h)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        hybridization_list = data.ligand_hybridization
        aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data.ligand_bond_feature = F.one_hot(data.ligand_bond_type.long() - 1, num_classes=len(utils_data.BOND_TYPES))
        return data


class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data
