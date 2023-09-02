from utils.evaluate import find_substructure
import sys, torch, os


if __name__ == '__main__':
    pt_path = sys.argv[1]
    smarts = sys.argv[2]
    for f_dir in os.listdir(pt_path): #logs_sampling/2023-09-01-22-07-00/sample-results/pt_files/
        pt_dir = os.path.join(pt_path, f_dir, 'sample_results', 'pt_files')
        pt_f = os.listdir(pt_dir)[0]
        pt_path_leaf = os.path.join(pt_dir, pt_f)
        ckp = torch.load(pt_path_leaf, map_location='cpu')
        smiles_list = ckp['smiles_list']
        sdf_path = ckp['sdf_path_list']
        for s, sdf in zip(smiles_list, sdf_path):
            for smart in ['c12ccccc1cncn2', 'c1ccccc1Nc2ncccn2']:
                if find_substructure(s, smarts):
                    print(s, smart, sdf)
    # ckp = torch.load(pt_path, map_location='cpu')
    # smiles_list = ckp['smiles_list']
    # sdf_path = ckp['sdf_path_list']
    # for s, sdf in zip(smiles_list, sdf_path):
    #     if find_substructure(s, smarts):
    #         print(s, sdf)

