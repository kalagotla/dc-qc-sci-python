# Import the libraries we need for this lab

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from IPython.display import clear_output

from .network import Net

torch.manual_seed(1)
dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0")


class Univariate:
    def __init__(self, layers=None, p_train=0.7, learning_rate=0.01, epochs=100, momentum=0.9):
        self.layers = layers
        if self.layers is None:
            self.layers = [1, 2, 5, 5, 2, 1]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.p_train = p_train
        self.data_set = Data()
        self.criterion = None
        self.net = None
        self.train_data_set, self.val_data_set = None, None
        self.train_loader = None
        self.optimizer = None

        pass

    def set_parameters(self):
        self.data_set = Data()
        self.criterion = torch.nn.MSELoss()
        self.net = Net(self.layers)
        self.train_data_set, self.val_data_set = random_split(self.data_set,
                                                              [int(np.round(self.p_train * self.data_set.__len__())),
                                                               int(np.round((1 - self.p_train) * self.data_set.__len__()))])
        self.train_loader = DataLoader(dataset=self.train_data_set, batch_size=64)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def live_plot(self, data, err_data, figsize=(14, 10), suptitle='', title1='', title2='', xlabel='', ylabel=''):
        clear_output(wait=True)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(suptitle)
        ax = fig.add_subplot(1, 2, 1)
        for xdata, ydata, label in data:
            ax.plot(xdata, ydata, label=label)
        ax.set_title(title1)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='center left')  # the plot evolves to the right
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

        # plot loss and R²
        ax = fig.add_subplot(1, 2, 2)
        for xdata, ydata, label in err_data:
            ax.plot(xdata, ydata, label=label)
        ax.set_title(title2)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc='center left')  # the plot evolves to the right
        ratio = 1.0
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        plt.show()

    # The function to calculate R² (coefficient of determination)

    def r_squared(self, model, data_set):
        yhat = model(data_set.x).detach()
        y = data_set.y
        ss_res = ((y - yhat) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return (1 - ss_res / ss_tot).item()

    def train(self, plot=True, plot_at=1, save_at=100, filename='../models/1var_model.tar'):
        LOSS = []
        R2 = []
        LOSST = []
        for epoch in range(self.epochs):
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                yhat = self.net(x)
                loss = self.criterion(yhat, y)
                loss.backward()
                self.optimizer.step()
            LOSS.append(loss.item())
            LOSST.append(sum(LOSS) / len(LOSS))
            R2.append(self.r_squared(self.net, self.data_set))

            if plot:
                if epoch % plot_at == 0:
                    plot_data = []
                    err_data = []
                    predicted = self.net(self.data_set.x).data.numpy()

                    plot_data.append([self.data_set.x.numpy(), self.data_set.y.numpy(), 'True data'])
                    plot_data.append([self.data_set.x.numpy(), predicted, 'Predictions'])
                    err_data.append([np.arange(len(R2)), R2, 'R² = ' + str(round(R2[-1], 4))])
                    err_data.append([np.arange(len(LOSST)), LOSST, 'Loss = ' + str(round(LOSST[-1], 4))])
                    self.live_plot(plot_data, err_data,
                                   suptitle='epoch = ' + str(epoch),
                                   title1='Function vs. DNN Model',
                                   title2='Loss and R²')

            if epoch % save_at == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.net.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,
                }, filename)

        return

    def restart(self):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.net.apply(weight_reset)

    def continue_train(self, filename='../models/1var_model.tar'):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def validation(self):
        val_loader = DataLoader(dataset=self.val_data_set, batch_size=64)

        a = []
        b = []
        for x, y in val_loader:
            a.append(x)
            b.append(y)

        val_x = a[0].numpy()
        old_val_x = val_x.copy()
        val_x = np.sort(val_x, axis=0)
        val_y = b[0].numpy()
        err_val_yhat = self.net(a[0]).data.numpy()
        val_yhat = self.net(torch.from_numpy(val_x)).data.numpy()

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(old_val_x, val_y, '.', label='dataset')
        ax.plot(val_x, val_yhat, label='predictions')
        ax.legend()

        val_accuracy = np.linalg.norm(val_y - err_val_yhat) ** 2 / len(val_y)
        print(f'Validation error for the trained model = {val_accuracy}')


class Data(Dataset):
    def __init__(self, n=100):
        self.x = torch.linspace(-1.5*np.pi, 2*np.pi, n, device=device, dtype=dtype)
        self.x = self.x.reshape(-1, 1)

        self.y = torch.sin(self.x) + torch.cos(self.x)
        self.y = self.y.reshape(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def plot(self, title='Dataset'):
        plt.figure()
        plt.plot(self.x, self.y, '.')
        plt.title(title)
        plt.show()
