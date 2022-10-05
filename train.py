import os
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from dataloader import FlowersDataset
import utils

# TODO: Data normalization (may require offline data augmentation); K-fold cross validation; Save log for visualization


class Trainer():
    def __init__(self, config_path, data_path, save_path, log_path):
        self.config = utils.load_config(config_path)
        self.save_path = os.path.join(save_path, f'{utils.current_time()}')
        self.log_path = os.path.join(log_path, f'{utils.current_time()}')
        self.train_meta_path = os.path.join(data_path, 'meta/train')
        self.eval_meta_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data = self.get_dataset()
        self.val_data = None
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"], drop_last=self.config["drop_last"])
        self.val_loader = None
        self.model = ResNet(self.config["model"]["in_channels"], eval(self.config["model"]["resblock"]), self.config["model"]["repeat"],
                            self.config["model"]["useBottleneck"], self.config["model"]["outputs"]).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.total_train_step = 0
        self.total_val_step = 0

    def get_dataset(self):
        # https://pytorch.org/vision/stable/transforms.html
        torch.manual_seed(7)
        transform1 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.RandomCrop([224, 224]),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor(),
            # Normalization
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
        transformed_dataset1 = FlowersDataset(
            meta_file='data/flowers/meta/train.txt', transform=transform1)
        torch.manual_seed(17)
        transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.RandomCrop([224, 224]),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
        transformed_dataset2 = FlowersDataset(
            meta_file='data/flowers/meta/train.txt', transform=transform2)
        torch.manual_seed(27)
        transform3 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([256, 256]),
            transforms.RandomCrop([224, 224]),
            transforms.ColorJitter(brightness=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])
        transformed_dataset3 = FlowersDataset(
            meta_file='data/flowers/meta/train.txt', transform=transform3)
        return ConcatDataset([transformed_dataset1, transformed_dataset2, transformed_dataset3])

    def train(self):
        self.model.train()
        size = len(self.train_loader.dataset)
        for batch, (img, label) in enumerate(self.train_loader):
            img, label = img.to(self.device), label.to(self.device)
            pred_label = self.model(img)
            loss = self.loss_fn(pred_label, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % self.config["print_loss_batches"] == 0:
                loss_val, curr = loss.item(), batch*len(img)
                print(f'loss: {loss_val:>7f}  [{curr:>5d}/{size:>5d}]')
            self.total_train_step += 1
            if self.total_train_step % self.config["save_log_steps"] == 0:
                # Save log
                pass

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            pass

    def start(self):
        utils.ensure_dir(self.log_path)
        utils.ensure_dir(self.save_path)
        print(f"Training on {self.device}...")
        for t in range(self.config["epochs"]):
            print(
                f'Epoch {t+1} ({utils.current_time()})\n-------------------------------')
            self.train()
            self.eval()
            if (t+1) % self.config["save_pth_epoches"] == 0:
                pth_path = os.path.join(self.save_path, f'{str(t+1)}.pth')
                torch.save(self.model.state_dict(), pth_path)
        pth_path = os.path.join(self.save_path, 'latest.pth')
        torch.save(self.model.state_dict(), pth_path)
        print(
            f'Completed {self.config["epochs"]} epoches; saved in "{self.save_path}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model trainer")
    parser.add_argument(
        '--config', type=str, default="configs/basic_config.json", help="Config path")
    parser.add_argument(
        '--data', type=str, default="data/flowers", help="Data directory")
    parser.add_argument(
        '--save', type=str, default='checkpoints', help="Checkpoint save directory")
    parser.add_argument(
        '--log', type=str, default="logs", help="Log directory")
    args = parser.parse_args()
    trainer = Trainer(args.config, args.data, args.save, args.log)
    trainer.start()