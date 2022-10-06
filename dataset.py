from importlib.util import set_loader
import torch
from torch.utils.data import Dataset
from torchvision import io


class FlowersDataset(Dataset):
    CLASSES = [
        'daffodil', 'snowdrop', 'lilyValley', 'bluebell', 'crocys', 'iris', 'tigerlily', 'tulip', 'fritillary',
        'sunflower', 'daisy', 'colts foot', 'dandelion', 'cowslip', 'buttercup', 'wind flower', 'pansy'
    ]

    def __init__(self, meta_file, transform=None):
        self.meta_file = meta_file
        self.data_paths, self.labels = self.read_meta(meta_file)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        image = io.read_image(self.data_paths[item])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[item]

    def read_meta(self, meta_file):
        data_paths = []
        labels = []
        f = open(meta_file, 'r')
        line = f.readline()
        while line:
            data_path, label = line.strip('\n').split(' ')
            data_paths.append(data_path)
            labels.append(int(label))
            line = f.readline()
        f.close()
        return data_paths, torch.tensor(labels)
