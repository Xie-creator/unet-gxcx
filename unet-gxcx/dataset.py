from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torchvision
from torchvision import transforms
import torch

class MyData(Dataset):
    def __init__(self, root_dir, image_dir, label_dir, transform=None):
        self.image_dir = image_dir;
        self.label_dir = label_dir;
        self.root_dir = root_dir;
        self.path_image = os.path.join(self.root_dir, self.image_dir)
        self.path_label = os.path.join(self.root_dir, self.label_dir)
        self.image_path = os.listdir(self.path_image)
        self.label_path = os.listdir(self.path_label)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_path[index]
        label_name = self.label_path[index]
        image_item_path = os.path.join(self.root_dir, self.image_dir, image_name)
        label_item_path = os.path.join(self.root_dir, self.label_dir, label_name)
        image = Image.open(image_item_path)
        label = Image.open(label_item_path)
        image = self.transform(image)
        label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.image_path)

#获取一个随机种子
seed = 1
torch.random.manual_seed(seed)
trans = transforms.Compose([transforms.ToTensor()]) #transforms.RandomCrop(256),

root_dir = r"G:\gxcx"
image_dir = r"wf"
label_dir = r"sim"

mydataset = MyData(root_dir, image_dir, label_dir, transform = trans)

train_loader = DataLoader(mydataset, batch_size=2, shuffle=True)
test_loader = DataLoader(mydataset, batch_size=1, shuffle=False)




