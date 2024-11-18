import torch
from torch import nn
from torchmetrics import Accuracy
import matplotlib.pyplot as plt
from tqdm import tqdm


class AcceleratorTrainer(nn.Module):
    def __init__(self, model, optimizer, loss_func, history_name):
        super(AcceleratorTrainer, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_func
        self.history = history_name

    def fit(self, epochs, dl_train, dl_val=None, device=torch.device('cuda:0')):
        accuracy_list = []
        # if torch.cuda.device_count() > 1:
        #     # print(f'Number of gpu is {torch.cuda.device_count()}')
        #     self.model = nn.DataParallel(self.model)
        self.model = self.model.to(device)

        save_path = f'{self.history}.pt'
        ckpt_path = f'{self.history}_ckpt.pt'

        for epoch in range(epochs):
            total_train = 0
            correct_sample = 0
            self.model.train()
            with tqdm(dl_train, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
                for batch, label in pbar:
                    batch, label = batch.to(device), label.to(device)
                    self.optimizer.zero_grad()
                    output = self.model(batch)
                    if isinstance(output, (tuple, list)):
                        """
                        multi-tasks train
                        """

                        output_spa = output[1]
                        output_fre = output[-1]
                        output = output[0]
                        loss = self.loss_fn(output, label) + self.loss_fn(output_spa, label) + self.loss_fn(output_fre, label)
                    else:
                        loss = self.loss_fn(output, label)
                    loss.backward()
                    self.optimizer.step()

                    total_train += label.shape[0]
                    _, predict = torch.max(output, 1)
                    correct_sample += (predict == label).sum().item()
                    train_acc = 100 * correct_sample / total_train
                    pbar.set_postfix({'train_accuracy': train_acc})

            epoch_acc = 100 * correct_sample / total_train
            if len(accuracy_list) >= 5:
                if epoch_acc >= accuracy_list[-1] or accuracy_list:
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f'epoch {epoch+1} reach the top accuracy, save_path: {ckpt_path}')
            accuracy_list.append(epoch_acc)
            epochs_list = range(0, epoch + 1)
            plt.plot(epochs_list, accuracy_list, 'b', marker='o')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.xlim(0, epochs)
            plt.legend(['Accuracy'])
            plt.savefig(f'{self.history}.png')

        torch.save(self.model.state_dict(), f'../model/Experiment/{save_path}')
        print(f'Training is finished , model save path:{save_path}')
