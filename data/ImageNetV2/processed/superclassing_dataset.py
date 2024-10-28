from imagenetv2_pytorch import ImageNetV2Dataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class CustomImageNetV2Dataset(ImageNetV2Dataset):
    def __init__(self, variant="matched-frequency", transform=None, location="."):
        super().__init__(variant=variant, transform=transform, location=location)
        if transform is None:
            self.transform = transforms.ToTensor()
        else:
            self.transform = transform
        self.labels = []
        self.superclass_labels = []
        for fname in self.fnames:
            label = int(fname.parent.name)
            self.labels.append(label)
            superclass_idx = label_to_superclass.get(label, superclass_to_idx["Other"])
            self.superclass_labels.append(superclass_idx)

    def __getitem__(self, idx):
        img_path = self.fnames[idx]
        img = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        superclass_idx = self.superclass_labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label, superclass_idx
