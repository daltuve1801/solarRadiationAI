import csv
from datetime import datetime
import numpy as np
from numpy.core.fromnumeric import size
import torch
from torch.utils.data import Dataset


class readPvData(Dataset):

    def __init__(self, rootpath, val=False):

        self.timestamp = []

        self.pvOutLabel = []

        gtFile = open(rootpath)  # annotations file
        csv_reader = csv.reader(gtFile, delimiter=',')
        next(csv_reader)

        for row in csv_reader:
            
            if len(row[0]) == 0:

                break

            dateEpochs = datetime.strptime(row[0], "%Y%m%d:%H%M")

            dateEpochs = datetime.timestamp(dateEpochs)

            self.timestamp.append(int(dateEpochs))

            self.pvOutLabel.append(float(row[1]))

        self.input = np.asarray(self.timestamp)
        self.input = np.transpose(self.input)
        #self.input = torch.Tensor(self.input).to(0)

        self.label = np.asarray(self.pvOutLabel)
        self.label = np.transpose(self.label)
        #self.label = torch.Tensor(self.label).to(0)

        #print(self.input.size())
        #print(self.label.size())


    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):

        x = self.input[idx]
        y = self.label[idx]

        sample = {'data': x, 'label': y}

        return sample
