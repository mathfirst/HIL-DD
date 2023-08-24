import torch
from torch.utils.data import Subset
from .pl_pair_dataset import PocketLigandPairDataset


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, split=config.split, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {
            k: [dataset.name2id[n] for n in names if n in dataset.name2id]
            for k, names in split_by_name.items()
        }
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset


def get_custom_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    split_file_path = config.split
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, split_file_path, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    return dataset

