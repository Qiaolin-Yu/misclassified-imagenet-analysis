import torch
import json
from PIL import Image
from imagenetv2_pytorch import ImageNetV2Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class SuperclassingImageNetDataset(ImageNetV2Dataset):
    def __init__(self, label_to_superclass, variant="matched-frequency", transform=None, location="."):
        super().__init__(variant=variant, transform=transform, location=location)
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.labels = []
        self.superclass_labels = []
        self.filtered_fnames = []
        self.label_to_superclass=label_to_superclass
        
        for fname in self.fnames:
            label = int(fname.parent.name)
            superclass_idx = self.label_to_superclass.get(label, None)
            if superclass_idx is not None:
                self.labels.append(label)
                self.superclass_labels.append(superclass_idx)
                self.filtered_fnames.append(fname)

    def __getitem__(self, idx):
        img_path = self.filtered_fnames[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        superclass_idx = self.superclass_labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, superclass_idx

    def __len__(self):
        return len(self.filtered_fnames)

