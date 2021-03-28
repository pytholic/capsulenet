import torch
import torchvision
from torchvision import datasets, transforms


class Dataset:
    def __init__(self, dataset, _batch_size, train_path, val_path):
        super(Dataset, self).__init__()
        if dataset == 'dementia':
            transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
            val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform)

            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
            self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=_batch_size, shuffle=True)
