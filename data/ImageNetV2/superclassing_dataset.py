import torch
import json
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class SuperclassImageNetV2Dataset(torch.utils.data.Dataset):
    def __init__(self, root, superclass=-1, transform=None, superclass_index_path='./superclass/superclass_index.json'):
        self.dataset = ImageNetV2Dataset(transform=transform, location=root)
        with open(superclass_index_path, 'r') as f:
            self.superclass_index = json.load(f)
        
        self.filtered_indices = [
            idx for idx, (_, label) in enumerate(self.dataset)
            if str(label) in self.superclass_index and (superclass == -1 or self.superclass_index[str(label)] == superclass)
        ]

    def __len__(self):
        return len(self.filtered_indices)

    def __getitem__(self, idx):
        original_idx = self.filtered_indices[idx]
        image, label = self.dataset[original_idx]
        superclass_label = self.superclass_index[str(label)]
        return image, label, superclass_label
    

class MappedSuperclassImageNetV2Dataset(SuperclassImageNetV2Dataset):
    def __init__(self, *args, label_mapping=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_mapping = label_mapping

    def __getitem__(self, index):
        inputs, label, super_label = super().__getitem__(index)
        if self.label_mapping is not None:
            label = self.label_mapping[label]
        return inputs, label, super_label
