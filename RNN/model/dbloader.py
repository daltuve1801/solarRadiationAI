import csv
from datetime import datetime
import numpy as np
from numpy.core.fromnumeric import size
import torch

def readPvData(rootpath):

    timestamp = []

    pvOutLabel = []

    gtFile = open(rootpath)  # annotations file
    csv_reader = csv.reader(gtFile, delimiter=',')
    next(csv_reader)

    for row in csv_reader:
        
        if len(row[0]) == 0:

            break

        dateEpochs = datetime.strptime(row[0], "%Y%m%d:%H%M")

        dateEpochs = datetime.timestamp(dateEpochs)

        timestamp.append(int(dateEpochs))

        pvOutLabel.append(float(row[1]))

    input = np.asarray(timestamp)
    input = np.transpose(input)
    #input = torch.Tensor(self.input).to(0)

    label = np.asarray(pvOutLabel)
    label = np.transpose(label)
    #label = torch.Tensor(self.label).to(0)

    #print(self.input.size())
    #print(self.label.size())

    return input,label