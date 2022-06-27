import matplotlib.pyplot as plt
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


class readTrafficSigns(Dataset):

    def __init__(self, rootpath, val=False):

        self.dates = []  # images
        self.labels = []  # corresponding labels
        if not val:
            for c in range(0, 43):
                # subdirectory for class
                prefix = rootpath + '/' + format(c, '05d') + '/'
                gtFile = open(prefix + 'GT-' + format(c, '05d') +
                              '.csv')  # annotations file
                # csv parser for annotations file
                gtReader = csv.reader(gtFile, delimiter=';')
                next(gtReader)  # skip header
                # loop over all images in current annotations file
                for row in gtReader:
                    img = plt.imread(prefix + row[0])
                   # imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    imgres = cv.resize(img, (64, 64))
                    imgnp = np.asarray(imgres)

                    self.images.append(imgnp)  # the 1th column is the filename
                    self.labels.append(row[7])  # the 8th column is the label
                gtFile.close()
        else:
            # subdirectory for class
            prefix = rootpath + '/'
            gtFile = open(prefix + 'GT-final_test.csv')  # annotations file
            # csv parser for annotations file
            gtReader = csv.reader(gtFile, delimiter=';')
            next(gtReader)  # skip header
            # loop over all images in current annotations file
            for row in gtReader:
                img = plt.imread(prefix + row[0])
               # imggray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                imgres = cv.resize(img, (64, 64))
                imgnp = np.asarray(imgres)

                self.images.append(imgnp)  # the 1th column is the filename
                self.labels.append(row[7])  # the 8th column is the label
            gtFile.close()

        self.images = np.asarray(self.images)
        self.images = torch.Tensor(self.images).to(0)
        self.images = torch.transpose(self.images, 1, 3).to(0)
        self.labels = np.asarray(self.labels)
        self.labels = self.labels.astype('str').reshape(-1, 1)
        self.labels = list(map(int, self.labels))
        self.labels = np.asarray(self.labels)
        self.labels = np.transpose(self.labels)
        self.labels = torch.Tensor(self.labels).long().to(0)
        print("Dataset cargado")
        print(self.images.shape)
        #self.labels = self.labels.squeeze()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.images[idx]
        y = self.labels[idx]

        sample = {'data': x, 'label': y}

        return sample


#if __name__ == "__main__":

    #sample = readTrafficSigns('GTSRB/Final_Test/Images', val=True)
