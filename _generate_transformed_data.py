import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from dataset import FlowersDataset
from PIL import Image

data_path = 'data/flowers'
train_meta_path = os.path.join(data_path, 'meta/train.txt')
transformed_img_path = os.path.join(data_path, 'transformed/imgs') 
transformed_meta_path = os.path.join(data_path, 'transformed/meta')

# https://pytorch.org/vision/stable/transforms.html
torch.manual_seed(7)
transform1 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([317, 317]),
    transforms.RandomRotation(degrees=45),
    transforms.CenterCrop([224,224]),
    transforms.ToTensor()
])
transformed_dataset1 = FlowersDataset(meta_file=train_meta_path, transform=transform1)

torch.manual_seed(17)
transform2 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([256, 256]),
    transforms.RandomCrop([224, 224]),
    transforms.ToTensor()
])
transformed_dataset2 = FlowersDataset(meta_file=train_meta_path, transform=transform2)

torch.manual_seed(27)
transform3 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor()
])
transformed_dataset3 = FlowersDataset(meta_file=train_meta_path, transform=transform3)

torch.manual_seed(37)
transform4 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([224, 224]),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.05),
    transforms.ToTensor()
])
transformed_dataset4 = FlowersDataset(meta_file=train_meta_path, transform=transform4)
transformed_dataset = ConcatDataset([transformed_dataset1, transformed_dataset2, transformed_dataset3, transformed_dataset4])
print(f'---Transformed {len(transformed_dataset)} images---')

# normalize the images
train_loader = DataLoader(transformed_dataset, batch_size=1)
channels_sum, channels_squared_sum = 0, 0
for batch, (img, label) in enumerate(train_loader):
    # Mean over batch, height and width, but not over the channels
    channels_sum += torch.mean(img, dim=[0,2,3])
    channels_squared_sum += torch.mean(img**2, dim=[0,2,3])
    
mean = channels_sum / len(transformed_dataset)

# std = sqrt(E[X^2] - (E[X])^2)
std = (channels_squared_sum / len(transformed_dataset) - mean ** 2) ** 0.5

print(f'---mean: {mean}---')
print(f'---std: {std}---')

normalization = transforms.Normalize(mean, std)
channels_sum = 0
for batch, (img, label) in enumerate(train_loader):
    img = normalization(img)
    channels_sum += torch.mean(img, dim=[0,2,3])
mean = channels_sum / len(transformed_dataset)

print(f'---Normalized {len(transformed_dataset)} images---')
print(f'---mean: {mean}---')

# save the images and generate the meta data
os.makedirs(transformed_img_path, exist_ok=True)
os.makedirs(transformed_meta_path, exist_ok=True)
transform = transforms.ToPILImage()

f = open(os.path.join(transformed_meta_path, 'train.txt'), 'w')
for batch, (img, label) in enumerate(train_loader):
    image = transform(img.reshape(3,224,224))
    image_path = f'{transformed_img_path}/image{batch}.jpg'
    image.save(image_path)
    line = image_path + ' ' + str(label.item()) + '\n'
    f.write(line)
f.close()

print(f'---Saved transformed {len(transformed_dataset)} images to {transformed_img_path}---')
print(f'---Saved meta for train data to {transformed_meta_path}---')
