from __future__ import print_function
import argparse
import torch
import torch.optim as optim
from Datasets import  ClassificationLoader
import numpy as np
from model_speech_yolo import VGG
from train_classification import train, test
import os
import sys



# Training settings
parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')
parser.add_argument('--train_path', default='gcommand_toy_example/train',
                    help='path to the train data folder')
parser.add_argument('--test_path', default='gcommand_toy_example/test',
                    help='path to the test data folder')
parser.add_argument('--valid_path', default='gcommand_toy_example/valid',
                    help='path to the valid data folder')
parser.add_argument('--batch_size', type=int, default=32,
                    metavar='N', help='training and valid batch size')
parser.add_argument('--test_batch_size', type=int, default=100,
                    metavar='N', help='batch size for testing')
parser.add_argument('--arc', default='VGG11',
                    help='network architecture: VGG11, VGG13, VGG16, VGG19')
parser.add_argument('--epochs', type=int, default=100,
                    metavar='N', help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001,
                    metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='SGD momentum, for SGD only')
parser.add_argument('--optimizer', default='adam',
                    help='optimization method: sgd | adam')
parser.add_argument('--cuda', default=True, help='enable CUDA')
parser.add_argument('--seed', type=int, default=1234,
                    metavar='S', help='random seed')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='num of batches to wait until logging train status')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')

# feature extraction options
parser.add_argument('--max_len', type=int, default=101,
                    help='window size for the stft')
parser.add_argument('--window_size', default=.02,
                    help='window size for the stft')
parser.add_argument('--window_stride', default=.01,
                    help='window stride for the stft')
parser.add_argument('--window_type', default='hamming',
                    help='window type for the stft')
parser.add_argument('--normalize', default=True,
                    help='boolean, wheather or not to normalize the spect')
parser.add_argument('--save_folder', type=str,  default='gcommand_pretraining_model/',
                    help='path to save the final model')
parser.add_argument('--class_num', type=int,  default=30,
                    help='number of classes to classify')


def build_model_name(args):
    args_dict = ["optimizer", "lr", "batch_size", "arc",  "class_num"]
    full_name = ""
    for arg in args_dict:
        full_name += str(arg) + "_" + str(getattr(args, arg)) + "_"

    return full_name[:-1] + ".pth"

args = parser.parse_args()
print(args)

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.manual_seed(args.seed)

# loading data
train_dataset = ClassificationLoader(args.train_path, window_size=args.window_size, window_stride=args.window_stride,
                                     window_type=args.window_type, normalize=args.normalize, max_len=args.max_len)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=20, pin_memory=args.cuda, sampler=None)

valid_dataset = ClassificationLoader(args.valid_path, window_size=args.window_size, window_stride=args.window_stride,
                                     window_type=args.window_type, normalize=args.normalize, max_len=args.max_len)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=None,
                                           num_workers=20, pin_memory=args.cuda, sampler=None)

test_dataset = ClassificationLoader(args.test_path, window_size=args.window_size, window_stride=args.window_stride,
                                    window_type=args.window_type, normalize=args.normalize, max_len=args.max_len)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=args.test_batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

# build model
if args.arc.startswith("VGG"):
    model = VGG(args.arc, args.class_num)
else:
    model = VGG("VGG11", args.class_num)

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

# define optimizer
if args.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer.lower() == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)
else:
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum)

best_valid_loss = np.inf
iteration = 0
epoch = 1


# trainint with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):
    train(train_loader, model, optimizer, epoch, args.cuda, args.log_interval)
    valid_loss, acc = test(valid_loader, model, args.cuda)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net':  model.module.state_dict() if args.cuda else model.state_dict(),
            'acc': valid_loss,
            'epoch': epoch,
            'class_num': args.class_num
        }
        if not os.path.isdir(args.save_folder):
            os.mkdir(args.save_folder)
        torch.save(state, args.save_folder + '/' + build_model_name(args))
    epoch += 1

# test model
test(test_loader, model, args.cuda)
