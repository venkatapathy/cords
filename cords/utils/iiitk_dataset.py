import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import PIL


class IIIT5K(Dataset):
    
    def __init__(self, root_dir, transform=None, train = True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        curr_dir=os.path.join(root_dir,'train' if train else 'test')
        file=open(os.path.join(root_dir,'train_Data.txt' if train else 'test_Data.txt'))
        self.files={os.path.join(curr_dir,line.strip().split(',')[0]):line.strip().split(',')[1] for line in file}
                 
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_name, label = list(self.files.items())[idx]
        image = PIL.Image.open(img_name).convert("RGB") # A few images are grayscale
        if self.transform:
            image = self.transform(image)
        sample = (image, label)
        return sample
    
def load_iiitk_dataset(root_dir):
    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = IIIT5K(root_dir=root_dir, transform=transform, train=True)
    #split train_dataset into train and validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = IIIT5K(root_dir=root_dir, transform=transform, train=False)
    return train_dataset,val_dataset, test_dataset