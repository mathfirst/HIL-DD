from utils.evaluate import find_substructure
import sys, torch


if __name__ == '__main__':
    pt_path = sys.argv[1]
    smarts = sys.argv[2]
    ckp = torch.load(pt_path, map_location='cpu')
    smiles_list = ckp['smiles_list']
    sdf_path = ckp['sdf_path_list']
    for s, sdf in zip(smiles_list, sdf_path):
        if find_substructure(s, smarts):
            print(s, sdf)

