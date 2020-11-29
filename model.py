from nyuv2 import NYUv2
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch;


dire="/root/depthEstima/Demo/Data_NYUV2"

t = transforms.Compose([transforms.RandomCrop(400), transforms.ToTensor()])
train= NYUv2(root=dire, download=True, 
       depth_transform=t,rgb_transform=t)


train_loader = DataLoader(train, batch_size=6, shuffle=True, num_workers=8, pin_memory=True)

for a,b in train_loader:

    pass