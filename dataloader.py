import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data

import os
import random
import numpy as np

TargetLabels = range(10)
PSeq1 = [{
    "breakpt": 1000,
    "distribution": {
        0: 0.1,
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        5: 0.1,
        6: 0.1,
        7: 0.1,
        8: 0.1,
        9: 0.1,
    },
}, {
    "breakpt": 2000,
    "distribution": {
        0: 0.1,
        2: 0.4,
        6: 0.2,
        7: 0.1,
        9: 0.2,
    },
}, {
    "breakpt": 3000,
    "distribution": {
        5: 0.2,
        6: 0.2,
        7: 0.2,
        8: 0.2,
        9: 0.2,
    },
}, {
    "breakpt": 4000,
    "distribution": {
        0: 0.1,
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        5: 0.1,
        6: 0.1,
        7: 0.1,
        8: 0.1,
        9: 0.1,
    },
}]
PSeq2 = [{
    "breakpt": 400,
    "distribution": {
        0: 0.1,
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        5: 0.1,
        6: 0.1,
        7: 0.1,
        8: 0.1,
        9: 0.1,
    },
}, {
    "breakpt": 800,
    "distribution": {
        0: 0.1,
        2: 0.4,
        6: 0.2,
        7: 0.1,
        9: 0.2,
    },
}, {
    "breakpt": 1200,
    "distribution": {
        5: 0.2,
        6: 0.2,
        7: 0.2,
        8: 0.2,
        9: 0.2,
    },
}]
PSeq3 = [{
    "breakpt": 400,
    "distribution": {
        0: 0.1,
        1: 0.1,
        2: 0.1,
        3: 0.1,
        4: 0.1,
        5: 0.1,
        6: 0.1,
        7: 0.1,
        8: 0.1,
        9: 0.1,
    },
}, {
    "breakpt": 800,
    "distribution": {
        0: 0.1,
        2: 0.4,
        6: 0.2,
        7: 0.1,
        9: 0.2,
    },
}, {
    "breakpt": 1200,
    "distribution": {
        5: 0.2,
        6: 0.2,
        7: 0.2,
        8: 0.2,
        9: 0.2,
    },
}]

PSeq = {
    'PSeq1': {
        'len': 5000,
        'seq': PSeq1,
    },
    'PSeq2': {
        'len': 1500,
        'seq': PSeq2,
    },
    'PSeq3': {
        'len': 1500,
        'seq': PSeq3,
    },
}


def get_indices(dataset, label):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == label:
            indices.append(i)
    return indices


random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def worker_init_fn(worker_id):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)


def raw_loaders(dataset_dir):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    raw_dataset = datasets.ImageFolder(dataset_dir, transform=preprocess)
    data_loaders = [
        iter(data.DataLoader(data.Subset(raw_dataset, get_indices(raw_dataset, i)),
                             batch_size=1, shuffle=True, worker_init_fn=worker_init_fn))
        for i in TargetLabels]
    return data_loaders


def load_with_probability_seq(seq_name, dataset_dir):
    seq_len = PSeq[seq_name]['len']
    seq = PSeq[seq_name]['seq']
    data_loaders = raw_loaders(dataset_dir)

    len_seq = len(seq)
    dis = [list(seq[k]['distribution'].items()) for k in range(len_seq)]
    breakpt = [seq[k]['breakpt'] for k in range(len_seq)]
    pi = 0

    for i in range(len_seq):
        cum = 0
        for j, (k, v) in enumerate(dis[i]):
            t = v
            dis[i][j] = (k, cum + v)
            cum += t

    for cnt in range(1, seq_len + 1):
        p = random.uniform(0, 1)
        if pi + 1 < len_seq and cnt >= breakpt[pi]:
            pi += 1

        target = 0
        for k, v in dis[pi]:
            if p <= v:
                target = k
                break

        out = next(data_loaders[target], None)
        if out is None:
            data_loaders = raw_loaders(dataset_dir)
            out = next(data_loaders[target], None)
        yield out


if __name__ == '__main__':
    data_path = './tiny-imagenet-200/train'

    ploader = load_with_probability_seq('PSeq1', data_path)
    for img, label in ploader:
        print(label)
