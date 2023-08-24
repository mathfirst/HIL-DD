import os, torch, pickle, lmdb
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file
from .pl_data import ProteinLigandData, torchify_dict
from torch_geometric.data import Data


class PocketLigandPairData(Data):
    def __init__(self, *args, **kwargs):
        super(PocketLigandPairData, self).__init__(*args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        else:
            return super().__inc__(key, value)


class PocketLigandPairDataset(Dataset):
    def __init__(self, raw_path, split=None, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None
        self.keys = None
        # self.name2id_path = None
        if not os.path.exists(self.processed_path):
            names_dict = torch.load(split)
            self.names = list(names_dict['train']) + list(names_dict['test'])
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None or (pocket_fn, ligand_fn) not in self.names: continue
                try:
                    pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    data = ProteinLigandData.from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                    )
                    data.protein_filename = pocket_fn
                    data.ligand_filename = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data.id = idx
        assert data.protein_pos.size(0) > 0
        # At the very first time, i.e. preparing .lmdb file, this if statement will not be executed.
        # if self.transform is not None and self.name2id_path is not None:
        if self.transform is not None:
            data = self.transform(data)
        return data


class PocketLigandPairDataset_(Dataset):
    # adapted from Jiaqi Guan's code
    def __init__(self, raw_path, split_file_path=None, transform=None, train=False, test=False):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        if train:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + '_processed_train.pt')
        elif test:
            self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                               os.path.basename(self.raw_path) + '_processed_test.pt')
        else:
            raise ValueError('Please specify either train or test is True.')
        self.transform = transform
        self.data_list = []
        self.train_data = []
        self.test_data = []
        if split_file_path is not None:
            split_dict = torch.load(split_file_path)
            if train:
                self.names = split_dict['train']
            else:
                self.names = split_dict['test']

        if not os.path.exists(self.processed_path):
            self._process()  # preprocessing data
            # torch.save(self.processed_data, self.processed_path)
        else:
            print('loading preprocessed pt')
            self.data_list = torch.load(self.processed_path)

            # self.train_names = split_dict['train']
            # self.test_names = split_dict['test']

    def _process(self):
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)  # 'index.pkl'

        num_skipped = 0
        # with db.begin(write=True, buffers=True) as txn:
        for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
            if pocket_fn is None or (pocket_fn, ligand_fn) not in self.names: continue
            # train = test = False
            # if (pocket_fn, ligand_fn) in test_names:
            #     test = True
            # elif (pocket_fn, ligand_fn) in train_names:
            #     train = True
            # else:
            #     print(f"{i} is not in split names.")
            #     continue
            try:
                pocket_dict = PDBProtein(os.path.join(self.raw_path, pocket_fn)).to_dict_atom()
                ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                data = PocketLigandPairData()
                for key, item in torchify_dict(pocket_dict).items():
                    data['protein_' + key] = item
                assert data.protein_pos.size(0) > 0, 'protein_pos.size(0) is not a positive integer.'
                for key, item in torchify_dict(ligand_dict).items():
                    data['ligand_' + key] = item
                data.protein_filename = pocket_fn
                data.ligand_filename = ligand_fn
                self.data_list.append(data)
                # if train:
                #     self.train_data.append(data)
                # elif test:
                #     self.test_data.append(data)
                # else:
                #     print("not train, not true, a bug")
                #     sys.exit()
                # self.processed_data[i] = data  # After Python 3.6, dict is ordered by default. Here is Python 3.8+.
            except:
                num_skipped += 1
                print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                continue

        torch.save(self.data_list, self.processed_path)

        # print(f"num_train: {len(self.train_data)}, num_test: {len(self.test_data)}")
        # print('saving training data to train_data.pt')
        # torch.save(self.train_data, 'train_data.pt')
        # print('saving test data to test_data.pt')
        # torch.save(self.test_data, 'test_data.pt')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        data.id = idx
        assert data.protein_pos.size(0) > 0, 'protein_pos.size(0) is not a positive integer.'
        # At the very first time, i.e. preparing .lmdb file, this if statement will not be executed.
        # When creating name2id file, we do not need to do data transformation.
        if self.transform is not None:  # and os.path.exists(self.name2id_path):
            data = self.transform(data)
        return data
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print(len(dataset), dataset[0])
