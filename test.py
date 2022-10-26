import os
import time
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, io
from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from dataset import FlowersDataset
import utils

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

class Tester():
    def __init__(self, config_path, data_path, save_path, plot):
        self.config = utils.load_config(config_path)
        self.save_path = save_path
        self.plot_confusion_matrix = plot
        self.test_meta_path = os.path.join(data_path, 'meta/test.txt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.test_data = FlowersDataset(meta_file=self.test_meta_path, transform=transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize([224, 224]), transforms.ToTensor(),
             transforms.Normalize(mean=[0.4493, 0.4431, 0.2898], std=[0.2810, 0.2535, 0.2626])]))
        self.test_loader = DataLoader(
            self.test_data, batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        self.model = ResNet(self.config["model"]["in_channels"], eval(self.config["model"]["resblock"]), self.config["model"]["repeat"],
                            self.config["model"]["useBottleneck"], self.config["model"]["outputs"]).to(self.device)

    def test(self):
        state_dict = torch.load(self.save_path)
        self.model.load_state_dict(state_dict)
        print(f'model {self.save_path} loaded')

        self.model.eval()
        size = len(self.test_loader.dataset)
        test_acc = 0.0
        labels = torch.IntTensor([]).to(self.device)
        preds = torch.IntTensor([]).to(self.device)
        with torch.no_grad():
            for img, label in self.test_loader:
                img, label = img.to(self.device), label.to(self.device)
                pred_vec = self.model(img)
                pred_label = pred_vec.argmax(dim=1)
                test_acc += (pred_label == label).float().sum()
                labels = torch.cat((labels, label), 0)
                preds = torch.cat((preds, pred_label), 0)

        test_acc /= size
        print(f"Test accuracy: {test_acc.item()}")

        if self.plot_confusion_matrix:
            sns.set()
            f, ax = plt.subplots()
            classes = []
            for i in range(17):
                classes.append(i)
            C2 = confusion_matrix(labels.tolist(), preds.tolist(), labels=classes)
            sns.heatmap(C2, annot=True, ax=ax) #画热力图

            ax.set_title('confusion matrix') #标题
            ax.set_xlabel('predict') #x轴
            ax.set_ylabel('true') #y轴

            plt.savefig('confusion_matrix/cm.png', format='png')
            print("cm saved")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Model tester")
    parser.add_argument(
        '--config', type=str, default="configs/basic_config_50.json", help="Config path")
    parser.add_argument(
        '--data', type=str, default="data/flowers", help="Data directory")
    parser.add_argument(
        '--save', type=str, default='checkpoints/2/34/100.pth', help="Checkpoint save directory")
    parser.add_argument(
        '--plot', type=bool, default=False, help="Whether to plot confusion matrix")
    args = parser.parse_args()
    tester = Tester(args.config, args.data, args.save, args.plot)
    tester.test()