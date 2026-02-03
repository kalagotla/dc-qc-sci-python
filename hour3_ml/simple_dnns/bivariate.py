# Import the libraries we need for this lab

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from IPython.display import clear_output
import time

torch.manual_seed(1)
dtype = torch.float
device = torch.device("cpu")


# device = torch.device("cuda:0")


class Bivariate:
    def __init__(self, layers=None, p_train=0.7, learning_rate=0.1, epochs=100, momentum=0.7):
        self.layers = layers
        if self.layers is None:
            self.layers = [2, 20, 20, 1]
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.p_train = p_train
        self.data_set = Data()
        self.net = None
        self.criterion = None
        self.train_data_set = None
        self.val_data_set = None
        self.train_loader = None
        self.optimizer = None
        pass

    def set_parameters(self):
        self.data_set = Data()
        self.criterion = torch.nn.MSELoss()
        self.net = Net(self.layers)
        self.train_data_set, self.val_data_set = self.data_set, self.data_set
        self.train_loader = DataLoader(dataset=self.train_data_set, batch_size=64)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=self.momentum)

    @staticmethod
    def live_plot_surf(self, data_set, data, err_data, figsize=(14, 9), suptitle='', title1='', title2='',
                       xlabel='', ylabel=''):
        clear_output(wait=True)
        fig = plt.figure(figsize=figsize)
        fig.suptitle(suptitle)
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        for xdata, ydata, label in data:
            ax.plot_surface(xdata[:, 0].reshape(data_set.x1len, data_set.x2len),
                            xdata[:, 1].reshape(data_set.x1len, data_set.x2len),
                            ydata.reshape(data_set.x1len, data_set.x2len), label=label)
        ax.set_title(title1)
        ax.grid(True)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.legend(loc='center left')  # the plot evolves to the right

        # plot loss and accuracy
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

    # The function to calculate the accuracy

    @staticmethod
    def accuracy(self, model, data_set):
        yhat = model(data_set.x)
        return (abs(yhat - data_set.y) <= 1e-1).numpy().mean()

    def train(self, plot=True, plot_at=1, save_at=100, filename='../models/2var_model.tar'):
        LOSS = []
        ACC = []
        LOSST = []
        for epoch in range(self.epochs):
            for x, y in self.train_loader:
                self.optimizer.zero_grad()
                yhat = self.net(x)
                loss = self.criterion(yhat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            LOSS.append(loss.item())
            LOSST.append(sum(LOSS) / len(LOSS))
            ACC.append(self.accuracy(self, self.net, self.data_set))

            if plot:
                if epoch % plot_at == 0:
                    plot_data = []
                    err_data = []
                    predicted = self.net(self.data_set.x).data.numpy()

                    plot_data.append([self.data_set.x.numpy(), self.data_set.y.numpy(), 'True data'])
                    plot_data.append([self.data_set.x.numpy(), predicted, 'Predictions'])
                    err_data.append([np.arange(len(ACC)), ACC, 'Accuracy = ' + str(ACC[-1])])
                    err_data.append([np.arange(len(LOSST)), LOSST, 'Total Loss = ' + str([LOSST[-1]])])
                    self.live_plot_surf(self, self.data_set, plot_data, err_data,
                                        suptitle='epoch = ' + str(epoch),
                                        title1='Function vs. DNN Model',
                                        title2='Loss and prediction accuracy')

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

    def continue_train(self, filename='../models/2var_model.tar'):
        checkpoint = torch.load(filename)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

    def validation(self):
        val_loader = DataLoader(dataset=self.val_data_set, batch_size=64)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(self.val_data_set.x[:, 0].reshape(self.val_data_set.x1len, self.val_data_set.x2len).numpy(),
                        self.val_data_set.x[:, 1].reshape(self.val_data_set.x1len, self.val_data_set.x2len).numpy(),
                        self.val_data_set.y.reshape(self.val_data_set.x1len, self.val_data_set.x2len).numpy())
        ax.plot_surface(self.val_data_set.x[:, 0].reshape(self.val_data_set.x1len, self.val_data_set.x2len).numpy(),
                        self.val_data_set.x[:, 1].reshape(self.val_data_set.x1len, self.val_data_set.x2len).numpy(),
                        self.net(self.val_data_set.x).data.reshape(self.val_data_set.x1len,
                                                                   self.val_data_set.x2len).numpy())

        val_err = np.linalg.norm(self.val_data_set.y.numpy() - self.net(self.val_data_set.x).data.numpy()) ** 2 \
                  / len(self.net(self.val_data_set.x).data)
        print(f'Validation error for the trained model = {val_err}')


class Data(Dataset):
    def __init__(self, n=100, m=100):
        x1 = torch.linspace(-1.5 * np.pi, 2 * np.pi, n, device=device)
        x2 = torch.linspace(-1.5 * np.pi, 2 * np.pi, m, device=device)

        x1, x2 = torch.meshgrid(x1, x2)

        y = torch.sin(x1) + torch.cos(x2)

        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
        self.x = torch.hstack((x1, x2))
        self.y = y.reshape(-1, 1)
        self.x1len = n
        self.x2len = m

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def plot(self, title='Bivariate data'):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.x[:, 0].reshape(self.x1len, self.x2len),
                        self.x[:, 1].reshape(self.x1len, self.x2len),
                        self.y.reshape(self.x1len, self.x2len))
        ax.set_title(title)
        ax.grid(True)


# Create the model class using relu as the activation function
# Create Net model class
class Net(nn.Module):

    # Constructor
    def __init__(self, Layers):
        super(Net, self).__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
            self.hidden.append(nn.Linear(input_size, output_size))

    # Prediction
    def forward(self, activation):
        L = len(self.hidden)
        for (l, linear_transform) in zip(range(L), self.hidden):
            if l < L - 1:
                activation = torch.tanh(linear_transform(activation))
            else:
                activation = linear_transform(activation)
        return activation
