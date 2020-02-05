__author__ = 'YaelSegal & TzeviyaFuchs'

import torch.optim as optim
import torch
import argparse
import Datasets
import numpy as np
import os
from model_speech_yolo import load_model, create_speech_model
from train_speech_yolo import train, test
import loss_speech_yolo

parser = argparse.ArgumentParser(description='train yolo model')
parser.add_argument('--train_data', type=str, default='librispeech_toy_example/train',
                    help='location of the train data')
parser.add_argument('--val_data', type=str, default='librispeech_toy_example/validation',
                    help='location of the validation data')
parser.add_argument('--arc', type=str, default='VGG19',
                    help='arch method (LeNet, VGG11, VGG13, VGG16, VGG19)')
parser.add_argument('--opt', type=str, default='adam',
                    help='optimization method: adam || sgd')
parser.add_argument('--momentum', type=float, default='0.9',
                    help='momentum')
parser.add_argument('--c_b_k', type=str, default='6_2_1000', help='C B K parameters')
parser.add_argument('--prev_classification_model', type=str, default='gcommand_pretraining_model/optimizer_adam_lr_0.001_batch_size_32_arc_VGG19_class_num_30.pth',
                    help='the location of the prev classification model')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float,  default=0.0,
                    help='dropout probability value')
parser.add_argument('--seed', type=int, default=1245,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--patience', type=int, default=5, metavar='N',
                    help='how many epochs of no loss improvement should we wait before stop training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--save_folder', type=str,  default='speechyolo_model',
                    help='path to save the final model')
parser.add_argument('--save_file', type=str,  default='',
                    help='filename to save the final model')
parser.add_argument('--trained_yolo_model', type=str, default='',
                    help='load model already trained by this script')
parser.add_argument('--augment_data', action='store_true', help='add data augmentation')
parser.add_argument('--noobject_conf',type=float, default=0.5,
                    help='noobject conf')
parser.add_argument('--obj_conf',type=float, default=1,
                    help='obj conf')
parser.add_argument('--coordinate',type=float, default=10,
                    help='coordinate')
parser.add_argument('--class_conf',type=float, default=1,
                    help='class_conf')
parser.add_argument('--loss_type',type=str, default="mse",
                    help='loss with abs or with mse (abs, mse)')
parser.add_argument('--decision_threshold',type=float, default=0.25,
                    help=' object exist threshold')
parser.add_argument('--iou_threshold',type=float, default=0.5,
                    help='high iou threshold')


def build_model_name(args):
    args_dict = ["opt", "lr", "batch_size", "arc", "c_b_k", "noobject_conf", "obj_conf",
                 "coordinate", "class_conf", "loss_type"]
    full_name = ""
    for arg in args_dict:
        if arg =="c_b_k":
            config_params = args.c_b_k.split('_')
            full_name += "c_{}_b_{}_k_{}".format(config_params[0], config_params[1], config_params[2])
            continue
        full_name += str(arg) + "_" + str(getattr(args, arg)) + "_"

    return full_name + ".pth"


args = parser.parse_args()
print(args)

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)


config_params = args.c_b_k.split('_')
param_names = ['C', 'B', 'K']  # C - cells, B - number of boxes per cell, K- number of keywords
config_dict = {i: int(j) for i, j in zip(param_names, config_params)}

# build model
if os.path.isfile(args.trained_yolo_model):  # model exists
    speech_net, check_acc, check_epoch = load_model(args.trained_yolo_model)
else:
    speech_net = create_speech_model(args.prev_classification_model, args.arc, config_dict, args.dropout)

if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    speech_net = torch.nn.DataParallel(speech_net).cuda()

# define optimizer
if args.opt.lower() == 'adam':
    optimizer = optim.Adam(speech_net.parameters(), lr=args.lr)
elif args.opt.lower() == 'sgd':
    optimizer = optim.SGD(speech_net.parameters(), lr=args.lr,
                          momentum=args.momentum)
else:
    optimizer = optim.SGD(speech_net.parameters(), lr=args.lr,
                          momentum=args.momentum)

train_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=args.train_data, this_root_dir=args.train_data,
                                            yolo_config=config_dict, augment = args.augment_data)
val_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=args.train_data, this_root_dir=args.val_data,
                                          yolo_config=config_dict)

sampler_train = Datasets.ImbalancedDatasetSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=False,
    num_workers=20, pin_memory=args.cuda, sampler=sampler_train)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=args.batch_size, shuffle=None,
    num_workers=20, pin_memory=args.cuda, sampler=None)

if os.path.isfile(args.trained_yolo_model):  # model exists
    best_valid_loss = check_acc
    epoch = check_epoch
    if args.cuda:
        best_valid_loss = best_valid_loss.cuda()
else:
    best_valid_loss = np.inf
    epoch = 1
iteration = 0


loss = loss_speech_yolo.YOLOLoss(noobject_conf= args.noobject_conf, obj_conf=args.obj_conf, coordinate=args.coordinate,
                                 class_conf=args.class_conf, loss_type=args.loss_type)

# training with early stopping
while (epoch < args.epochs + 1) and (iteration < args.patience):

    if epoch < 5 and args.opt == 'sgd':
        optimizer = optim.Adam(speech_net.parameters(), lr=args.lr)
    elif epoch == 5 and args.opt == 'sgd':
        optimizer = optim.SGD(speech_net.parameters(), lr=args.lr, momentum=args.momentum)

    train(train_loader, speech_net, loss.loss, config_dict, optimizer, epoch, args.cuda, args.log_interval)
    valid_loss = test(val_loader, speech_net, loss.loss, config_dict, args.decision_threshold, args.iou_threshold, args.cuda)
    if valid_loss > best_valid_loss:
        iteration += 1
        print('Loss was not improved, iteration {0}'.format(str(iteration)))
    else:
        print('Saving model...')
        iteration = 0
        best_valid_loss = valid_loss
        state = {
            'net': speech_net.module.state_dict() if args.cuda else speech_net.state_dict(),
            'acc': valid_loss,
            'epoch': epoch,
            'config_dict': config_dict,
            'arc': args.arc,
            'dropout': args.dropout,
            'loss_params': {"noobject_conf": args.noobject_conf, "obj_conf": args.obj_conf,
                            "coordinate": args.coordinate, "class_conf": args.class_conf, "loss_type": args.loss_type}
        }
        if not os.path.isdir(args.save_folder):
            os.mkdir(args.save_folder)

        if args.save_file:
            torch.save(state, args.save_folder + '/' + args.save_file)
        else:
            torch.save(state, args.save_folder + '/' + build_model_name(args))
    epoch += 1
