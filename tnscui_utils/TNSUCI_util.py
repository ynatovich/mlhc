from sklearn.model_selection import StratifiedKFold
import os
import warnings
import random
import numpy as np
import csv

warnings.filterwarnings('ignore')
sep = os.sep
filesep = sep  # set separator


def char_color(s,front=50,word=32):
    """
    # Function to change the color of a string
    :param s: 
    :param front: 
    :param word: 
    :return: 
    """
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char

def array_shuffle(x,axis = 0, random_state = 2020):
    """
    For multidimensional arrays, shuffle on any axis
    :param x: ndarray
    :param axis: scrambled axis
    :return:shuffled array
    """
    new_index = list(range(x.shape[axis]))
    random.seed(random_state)
    random.shuffle(new_index)
    x_new = np.transpose(x, ([axis]+[i for i in list(range(len(x.shape))) if i is not axis]))
    x_new = x_new[new_index][:]
    new_dim = list(np.array(range(axis))+1)+[0]+list(np.array(range(len(x.shape)-axis-1))+axis+1)
    x_new = np.transpose(x_new, tuple(new_dim))
    return x_new

def get_filelist_frompath(filepath, expname, sample_id = None):
    """
    Read files with a fixed extension in a folder
    :param filepath:
    :param expname: Extension, such as 'h5', 'PNG'
    :param sample_id: You can only read pictures with fixed patient ids
    :return: file path list
    """
    file_name = os.listdir(filepath)
    file_List = []
    if sample_id is not None:
        for file in file_name:
            if file.endswith('.'+expname):
                id = int(file.split('.')[0])
                if id in sample_id:
                    file_List.append(os.path.join(filepath, file))
    else:
        for file in file_name:
            if file.endswith('.'+expname):
                file_List.append(os.path.join(filepath, file))
    return file_List

def readCsv(csvfname):
    # read csv to list of lists
    with open(csvfname, 'r') as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_fold_filelist(csv_file, K=3, fold=1, random_state=2020, validation=False, validation_r = 0.2):
    """
    API to obtain the results of the analysis (based on the size of the 3-tier category balance analysis)
    :param csv_file: File with ID, CATE, size
    :param K: Folds
    :param fold: Return the number of folds, starting from 1
    :param random_state: Random number seed, after being fixed, the splitting is the same for each experiment (note that switching versions of sklearn may cause different splitting results for the same random number seed)
    :param validation:Do you need a validation set (randomly sample part of the data from the training set as a validation set)
    :param validation_r: Extract the proportion of the validation set to the training set
    :return: A list of train and test, with label and size
    """

    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    print('header', header)
    nodules = CTcsvlines[1:]

    # Extract t
    sizeall = [int(i[2]) for i in nodules]
    sizeall.sort()
    low_mid_thre = sizeall[int(len(sizeall)*1/3)]
    mid_high_thre = sizeall[int(len(sizeall)*2/3)]

    # Divided into three groups of low, mid, and high according to the size tertile

    low_size_list = [i for i in nodules if int(i[2]) < low_mid_thre]
    mid_size_list = [i for i in nodules if int(i[2]) < mid_high_thre and int(i[2]) >= low_mid_thre]
    high_size_list = [i for i in nodules if int(i[2]) >= mid_high_thre]

    low_label = [int(i[1]) for i in low_size_list]
    mid_label = [int(i[1]) for i in mid_size_list]
    high_label = [int(i[1]) for i in high_size_list]


    low_fold_train = []
    low_fold_test = []

    mid_fold_train = []
    mid_fold_test = []

    high_fold_train = []
    high_fold_test = []

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(low_label, low_label):
        low_fold_train.append([low_size_list[i] for i in train])
        low_fold_test.append([low_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(mid_label, mid_label):
        mid_fold_train.append([mid_size_list[i] for i in train])
        mid_fold_test.append([mid_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(high_label, high_label):
        high_fold_train.append([high_size_list[i] for i in train])
        high_fold_test.append([high_size_list[i] for i in test])

    if validation is False: # If no verification set is set, return directly
        train_set = low_fold_train[fold-1]+mid_fold_train[fold-1]+high_fold_train[fold-1]
        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]
        return [train_set, test_set]
    else:  #Set the verification set, and then draw a certain number of samples from the training set "category and size balanced" as the verification set
        # Separate the positive and negative samples of each size layer in the fold
        low_fold_train_p = [i for i in low_fold_train[fold-1] if int(i[1]) == 1]
        low_fold_train_n = [i for i in low_fold_train[fold-1] if int(i[1]) == 0]

        mid_fold_train_p = [i for i in mid_fold_train[fold-1] if int(i[1]) == 1]
        mid_fold_train_n = [i for i in mid_fold_train[fold-1] if int(i[1]) == 0]

        high_fold_train_p = [i for i in high_fold_train[fold-1] if int(i[1]) == 1]
        high_fold_train_n = [i for i in high_fold_train[fold-1] if int(i[1]) == 0]

        # Extract the verification set of each size layer and combine
        validation_set = low_fold_train_p[0:int(len(low_fold_train_p) * validation_r)] + \
                         low_fold_train_n[0:int(len(low_fold_train_n) * validation_r)] + \
                         mid_fold_train_p[0:int(len(mid_fold_train_p) * validation_r)] + \
                         mid_fold_train_n[0:int(len(mid_fold_train_n) * validation_r)] + \
                         high_fold_train_p[0:int(len(high_fold_train_p) * validation_r)] + \
                         high_fold_train_n[0:int(len(high_fold_train_n) * validation_r)]

        # Extract the training set of each size layer and combine
        train_set = low_fold_train_p[int(len(low_fold_train_p) * validation_r):] + \
                         low_fold_train_n[int(len(low_fold_train_n) * validation_r):] + \
                         mid_fold_train_p[int(len(mid_fold_train_p) * validation_r):] + \
                         mid_fold_train_n[int(len(mid_fold_train_n) * validation_r):] + \
                         high_fold_train_p[int(len(high_fold_train_p) * validation_r):] + \
                         high_fold_train_n[int(len(high_fold_train_n) * validation_r):]

        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]

        return [train_set, validation_set, test_set]




from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau

class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

