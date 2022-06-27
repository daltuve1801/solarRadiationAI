
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from sklearn import metrics
import dataset
from torch.utils.data import Dataset, DataLoader, random_split


class model(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, lr, lossFn, activation, dp=0.4):


        super(model, self).__init__()
        self.lossFn = lossFn
        self.lr = lr
        self.activation = activation
        self.cnn_layers = torch.nn.Sequential(
            # Defining a 2D convolution layer
            torch.nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(1024, 43)
        )

    def setOptimizer(self, opt):
        self.optimizer = opt

    def forward(self, X):
    
        y_out = self.cnn_layers(X)
        y_out = y_out.view(y_out.size(0), -1)
        y_out = self.linear_layers(y_out)
        return y_out

    def backward(self, y_out, y, train=True):
        lossFn = self.lossFn(y_out, y)
        if(train):
            self.optimizer.zero_grad()
            lossFn.backward()
            self.optimizer.step()
        return lossFn

    def metricsc(self, y_out, y):

        f1_score = 0
        y_p = np.argmax(y_out, axis=1)
        M_confusion = metrics.confusion_matrix(y, y_p)
        f1_score = metrics.f1_score(y, y_p, average='micro')
        return M_confusion, f1_score


def save(model, epoca, f1_score, cost):
    torch.save({
        'epoca': epoca,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': model.optimizer.state_dict(),
        'cost': cost,
        'f1_score': f1_score,
    }, "best_model.pth")

    f = open("best_data.txt", "w")
    f.write("Epoca: " + str(epoca) +
            "   Cost: {:.5f} ".format(cost)+"f1 {:.5f}".format(f1_score))
    f.close()


def plot(cost):

    # Cost fuction'
    plt.plot(cost, color='cyan')
    plt.ylabel('Cost fuction')
    plt.xlabel('epochs')
    plt.legend(title='Cost Function vs Epochs')
    plt.title('Cost Function vs Epochs')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Dataset
    trafficDataset = dataset.readTrafficSigns('Final_Training/Images')
    loader_train = DataLoader(trafficDataset, batch_size=1024, shuffle=True)
    print(loader_train)
    trafficDataset_test = dataset.readTrafficSigns(
        'Final_Test/Images', val=True)
    loader_val = DataLoader(trafficDataset_test,
                            batch_size=1024, shuffle=False)
   
    # Definición de la red
    lr = 0.0001
    epochs = 100
    lossFn = torch.nn.CrossEntropyLoss().to(0)
    activation = torch.nn.Tanh().to(0)
    model = model(10000, 43, lr, lossFn, activation).to(0)
    model.setOptimizer(torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.005))

    train_cost = []
    val_cost = []
    best_f1 = 0
    for i in range(epochs):

        # Train
        model.train()
        cost = 0
        full_label = []
        full_y = []
        for batch_idx, data in enumerate(loader_train, 0):
            
            y_out = model.forward(data["data"])
            cost += model.backward(y_out, data['label'])
            full_y = np.append(
                full_y, y_out.cpu().detach().numpy().squeeze())
            full_label = np.append(
                full_label, data['label'].cpu().detach().numpy().squeeze())
        full_y = full_y.reshape(39209, -1)
        full_label = full_label.reshape(39209, -1)
        M_confusion, f1_score = model.metricsc(full_y, full_label)
        cost = cost / (batch_idx+1)
        train_cost.append(cost.cpu().detach().numpy().squeeze())
        print("Train")
        print("Epoch: " + str(i+1) +
              "   Cost: {:.5f}".format(cost) + "  F1: {:.5f}".format(f1_score))
        # Eval model
        with torch.no_grad():
            model.eval()
            cost = 0
            full_label = []
            full_y = []
            for batch_idx, data in enumerate(loader_val, 0):
                y_out = model.forward(data["data"])
                cost += model.backward(y_out, data['label'], False)
                full_y = np.append(
                    full_y, y_out.cpu().detach().numpy().squeeze())
                full_label = np.append(
                    full_label, data['label'].cpu().detach().numpy().squeeze())
            full_y = full_y.reshape(12630, -1)
            full_label = full_label.reshape(12630, -1)
            M_confusionv, f1_score = model.metricsc(full_y, full_label)
            cost = cost / (batch_idx+1)
            val_cost.append(cost.cpu().detach().numpy().squeeze())
            print("eval")
            print("Epoch: " + str(i+1) +
                  "   Cost: {:.5f}".format(cost) + "  F1: {:.5f}".format(f1_score))

            if (f1_score > best_f1):
                print("Mejor")
                best_f1 = f1_score
                save(model, i, f1_score, cost)
                # print(M_confusionv)
                # print(M_confusion)


    plot(train_cost)
    plot(val_cost)
