import sys,os
curPath = os.path.abspath(os.path.dirname(__file__))
parentPath = os.path.split(curPath)[0]
rootPath = os.path.split(parentPath)[0]
sys.path.append(rootPath)
sys.path.append(rootPath+"/cnn-gen")

# import numpy as np
from random import random

# from __future__ import print_function

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

import evaluate_torch
import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Cutout data enhance
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
        	# (x,y)表示方形补丁的中心位置
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# Data enhance
cifar_norm_mean = (0.49139968, 0.48215827, 0.44653124)
cifar_norm_std = (0.24703233, 0.24348505, 0.26158768)

'''
# Data enhance without cutout
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])
'''

# Data enhance with cutout
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
    Cutout(n_holes=1, length=16)
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_norm_mean, cifar_norm_std),
])

torch_batch_size = 100

gpu = False

first_time = True

best_on_test_set = 0

use_sgdr = True

#net_dict = {}

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=torch_batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=torch_batch_size,
                                         shuffle=False, num_workers=2)

'''
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
'''

# 第epoch值进行计算并更新学习率
def get_adjusted_lr(epoch, T_0=5, T_mult=2, eta_max=0.1, eta_min=0.):
    i = np.log2(epoch / T_0 + 1).astype(np.int)
    T_cur = epoch - T_0 * (T_mult ** (i) - 1)
    T_i = (T_0 * T_mult ** i)
    cur_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * T_cur / T_i))
    return cur_lr

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_layer_param(model):
    return sum([torch.numel(param) for param in model.parameters()])

# evaluate the fitness
# batch_size = 0 means evaluate until reach the end of the evaluate set
def eval_fitness(net, loader, batch_size, torch_batch_size, start, gpu):

    # eval() only changes the state of some modules, e.g., dropout, but do not disable loss back-propogation.
    # By setting eval(), dropout() does not work and is temporarily removed from the chain of update.

    #switch to evaluation mode
    net.eval()

    hit_count = 0
    total = 0

    i = 0
    with torch.no_grad():
        for num, data in enumerate(loader, start):
            i += 1
            total += torch_batch_size

            # 得到输入数据
            inputs, labels = data

            # 包装数据
            if gpu:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            else:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
            try:

                outputs = net(inputs)
                _, predicted = outputs.max(1)
                hit_count += predicted.eq(labels).sum()

            except Exception as e:
                print(e)

            if (i == batch_size): #because i > 0 then no need to judge batch_size != 0
                break

    #switch to training mode
    net.train()

    return (hit_count.item() / total)

def train(config, use_sgdr):

    global gpu
    global first_time
    global best_on_test_set

    best_every_generation = list()

    if torch.cuda.is_available():
        gpu = True
        print("Running on GPU!")
    else:
        gpu = False
        print("Running on CPU!")

    if use_sgdr:
        max_epoch = config.sgdr_epoch
    else:
        max_epoch = config.max_epoch
    print(max_epoch)

    ep = open("epoch.csv", "a")
    ep.write("init_channel_num, {0:d}, num_layer, {1:d}, num_dense_layer, {2:d}\n".format(config.init_channel_num, config.num_layer, config.num_dense_layer))
    ep.close()

    if gpu:
        net.cuda()

    net.train()

    criterion = nn.CrossEntropyLoss()  # use a Classification Cross-Entropy loss
    optimizer = optim.SGD(net.parameters(), config.init_learning_rate, momentum=0.9, weight_decay=5e-4)

    # Evalute the fitness before trainning
    evaluate_batch_size = 0
    start = 0

    fit = eval_fitness(net, testloader, evaluate_batch_size, torch_batch_size, start, gpu)

    #comp = open("comp.csv", "a")
    #comp.write('{0},{1:3.3f},'.format(j, fit))
    print('Before: {0:3.5f}'.format(fit))

    # train the network
    epoch = 0
    lr_reduce = True
    lr_total_reduce_times = 3  # times lr reduce to its 0.1
    lr_reduce_times = 0  # current times lr reduced

    precision_count = 0
    precision_count_max = 5
    best_precision = 0.0

    evaluate_and_print_interval = 10

    training = True
    train_epoch = max_epoch
    lr = config.init_learning_rate
    while training and epoch < train_epoch:  # loop over the dataset multiple times
        # for epoch in range(10):
        net.train()
        epoch += 1
        running_loss = 0.0
        correct = 0
        total = 0

        if use_sgdr:
            cur_lr = get_adjusted_lr(epoch = epoch, T_0=config.sgdr_t0, T_mult=config.sgdr_t_mult, eta_max=config.init_learning_rate)
            optimizer = optim.SGD(net.parameters(), cur_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
            print('Epoch {0}: {1:1.8f}'.format(epoch, cur_lr))
        else:
            print('Epoch {0}: {1:1.8f}'.format(epoch, lr))

        if not(use_sgdr) and (train_epoch > lr_total_reduce_times) and (epoch % (train_epoch // lr_total_reduce_times) == 0):
            lr /= 10
            optimizer = optim.SGD(net.parameters(), lr, momentum=0.9)
            print("Learning rate set to: {0:1.8f}".format(lr))
        

        mixup = True # If use mixup or not

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if gpu:
                inputs, labels = inputs.to("cuda"), labels.to("cuda")
            else:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")

            # Mixup
            if mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 1.)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            if mixup:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # record the losses
            running_loss += loss.data.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            # num_loss += 1

            # print statistics
            if i % 100 == 0:  # print every 100 mini-batches
                print(i, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (running_loss / (i + 1), 100. * correct / total, correct, total))


        # print precision every epoch

        #if epoch % evaluate_and_print_interval == (evaluate_and_print_interval - 1):
        fitness_test = eval_fitness(net, testloader, 0, torch_batch_size, 0, gpu)

        #save the best net on test set
        if fitness_test > best_on_test_set:
            best_on_test_set = fitness_test
            torch.save(net, "best.pkl")

        print('Epoch {1:d}: {0:3.5f}'.format(fitness_test, epoch))
        ep = open("epoch.csv", "a")
        if use_sgdr:
            ep.write("{0:d}, {1:3.5f}, {2:3.8f}\n".format(epoch, fitness_test, cur_lr))
        else:
            ep.write("{0:d}, {1:3.5f}, {2:3.8f}\n".format(epoch, fitness_test, lr))
        ep.close()
        # reload run parameters

    print('Finished Training')

    fitness_train = eval_fitness(net, trainloader, 0, torch_batch_size, 0, gpu)
    fitness_test = eval_fitness(net, testloader, 0, torch_batch_size, 0, gpu)

    print('After: {0:3.5f}, {1:3.5f}\n'.format(fitness_train, fitness_test))

    torch.save(net, str(config.init_channel_num)+"-"+str(config.num_layer)+"-"+"final.pkl")

# Load configuration.
config = config.Config('config-cnn-gen')

para = [(8, 15),  (8, 20),  (8, 25),  (8, 30),
        (16, 15), (16, 20), (16, 25), (16, 30),
        (24, 15), (24, 20), (24, 25), (24, 30),
        (32, 15), (32, 20), (32, 25), (32, 30)]

while len(para):
    channel, layer = para.pop(0)
    config.init_channel_num = channel
    config.num_layer = layer
    config.num_cnn_layer = config.num_layer - 2
    net = evaluate_torch.Net(config)
    train(config, True)
    train(config, True)

'''
if load_from_file:
    net = torch.load("final.pkl")
else:
'''