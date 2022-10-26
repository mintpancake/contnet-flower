import os
import time
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms, io
from models.resnet import ResNet, ResBlock, ResBottleneckBlock
from dataset import FlowersDataset
import utils
from torchmetrics import F1Score   
from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, config_path, data_path, save_path, log_path):
        self.config = utils.load_config(config_path)
        self.save_path = os.path.join(save_path, f'{utils.current_time()}')
        self.loss_log_path = os.path.join(
            log_path, 'loss/', f'{utils.current_time()}')
        self.acc_log_path = os.path.join(
            log_path, 'acc/', f'{utils.current_time()}')
        self.f1_log_path = os.path.join(
            log_path, 'f1/', f'{utils.current_time()}')
        self.train_meta_path = os.path.join(
            data_path, 'transformed/meta/train.txt')
        self.eval_meta_path = os.path.join(data_path, 'meta/val.txt')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data = FlowersDataset(meta_file=self.train_meta_path, transform=transforms.Compose(
            [transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=[0.4493, 0.4431, 0.2898], std=[0.2810, 0.2535, 0.2626])]))
        self.val_data = FlowersDataset(meta_file=self.eval_meta_path, transform=transforms.Compose(
            [transforms.ToPILImage(), transforms.Resize([224, 224]), transforms.ToTensor(),
             transforms.Normalize(mean=[0.4493, 0.4431, 0.2898], std=[0.2810, 0.2535, 0.2626])]))
        self.train_loader = DataLoader(
            self.train_data, batch_size=self.config["batch_size"], shuffle=self.config["shuffle"], drop_last=self.config["drop_last"])
        self.val_loader = DataLoader(
            self.val_data, batch_size=self.config["batch_size"], shuffle=False, drop_last=False)
        self.model = ResNet(self.config["model"]["in_channels"], eval(self.config["model"]["resblock"]), self.config["model"]["repeat"],
                            self.config["model"]["useBottleneck"], self.config["model"]["outputs"]).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.config["learning_rate"], weight_decay=self.config["weight_decay"])
        self.total_train_step = 0
        self.total_val_step = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.loss_writer = SummaryWriter(self.loss_log_path)
        self.acc_writer = SummaryWriter(self.acc_log_path)
        self.f1_writer = SummaryWriter(self.f1_log_path)

    def train(self):
        self.model.train()
        size = len(self.train_loader.dataset)
        for batch, (img, label) in enumerate(self.train_loader):
            img, label = img.to(self.device), label.to(self.device)
            pred_vec = self.model(img)
            loss = self.loss_fn(pred_vec, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if batch % self.config["print_loss_batches"] == 0:
                loss_val, curr = loss.item(), batch*len(img)
                print(f'loss: {loss_val:>7f}  [{curr:>5d}/{size:>5d}]')
            self.total_train_step += 1
            # save the loss info of steps to loss log file
            if self.total_train_step % self.config["save_loss_log_steps"] == 0:
                self.loss_writer.add_scalar(
                    'Training loss', loss.item(), self.total_train_step)
                pass

    def eval(self):
        self.model.eval()
        size = len(self.val_loader.dataset)
        loss_fn = nn.CrossEntropyLoss(reduction='sum').to(self.device)
        val_loss = 0.0
        val_acc = 0.0
        labels = torch.IntTensor([]).to(self.device)
        preds = torch.IntTensor([]).to(self.device)
        with torch.no_grad():
            for img, label in self.val_loader:
                img, label = img.to(self.device), label.to(self.device)
                pred_vec = self.model(img)
                loss = loss_fn(pred_vec, label)
                val_loss += loss
                pred_label = pred_vec.argmax(dim=1)
                val_acc += (pred_label == label).float().sum()
                labels = torch.cat((labels, label), 0)
                preds = torch.cat((preds, pred_label), 0)

        val_loss /= size
        val_acc /= size

        # calculate F1 score
        f1_score = F1Score(num_classes=17, average='macro').to(self.device)
        f1 = f1_score(preds, labels)

        self.end_time = time.time()
        print(f'Test error: \n'
              f'  Avg loss: {val_loss:>8f} \n'
              f'  Avg accu: {val_acc:>8f} \n'
              f'  F1 score: {f1:>8f} \n'
              f'      Time: {(self.end_time - self.start_time):>8f} \n')
        self.total_val_step += 1
        # save the accuracy info of epochs to acc log file
        if self.total_val_step % self.config["save_acc_log_steps"] == 0:
            self.acc_writer.add_scalar(
                'Training accuracy', val_acc, self.total_val_step)

        # save the F1 score to f1 log file
        if self.total_val_step % self.config["save_f1_log_steps"] == 0:
            self.f1_writer.add_scalar(
                'F1 score', f1, self.total_val_step)

    def start(self):
        utils.ensure_dir(self.loss_log_path)
        utils.ensure_dir(self.acc_log_path)
        utils.ensure_dir(self.save_path)
        print(f"Training on {self.device}...")
        self.start_time = time.time()
        for t in range(self.config["epochs"]):
            print(
                f'Epoch {t+1} ({utils.current_time()})\n-----------------------------')
            self.train()
            self.eval()
            if (t+1) % self.config["save_pth_epochs"] == 0:
                pth_path = os.path.join(self.save_path, f'{str(t+1)}.pth')
                torch.save(self.model.state_dict(), pth_path)
        pth_path = os.path.join(self.save_path, 'latest.pth')
        torch.save(self.model.state_dict(), pth_path)
        self.loss_writer.close()
        self.acc_writer.close()
        self.f1_writer.close()
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
