__author__ = 'YaelSegal & TzeviyaFuchs'

import torch.optim as optim
import torch
import argparse
import Datasets
import numpy as np
import time
import os
import pdb
from model_speech_yolo import load_model
import train_speech_yolo
import loss_speech_yolo

parser = argparse.ArgumentParser(description='test  speechYolo model')
parser.add_argument('--train_data', type=str,
                    default='librispeech_toy_example/train',
                    help='location of the train data')
parser.add_argument('--test_data', type=str,
                    default='librispeech_toy_example/test',
                    help='location of the train data')
parser.add_argument('--word_list', type=str,
                    default='word_list.txt',
                    help='location of the word list ')
parser.add_argument('--model', type=str,
                    default='speechyolo_model/opt_adam_lr_0.001_batch_size_32_arc_VGG19_c_6_b_2_k_1000noobject_conf_0.5_obj_conf_1_coordinate_10_class_conf_1_loss_type_mse_.pth',
                    help='the location of the trained speech yolo model')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1245,
                    help='random seed')
parser.add_argument('--theta_range', type=str, default='0.1_1.0_0.1', help='0.0_1.0_0.1 format: from 0.0 to 1.0 with step of 0.1')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--iou_threshold', type=float, default=0.5,
                    help='high iou threshold')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


model, acc, epoch = load_model(args.model)
config_dict = {"C": model.c, "B": model.b, "K": model.k}
if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

test_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=args.train_data, this_root_dir=args.test_data,
                                          yolo_config=config_dict, words_list_file=args.word_list)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20,
                                          pin_memory=args.cuda, sampler=None)

range_split = args.theta_range.split('_')
start_theta = float(range_split[0])
end_theta = float(range_split[1])
step_theta = float(range_split[2])

loss = loss_speech_yolo.YOLOLoss()
# threshold = args.decision_threshold
atwv_dict = {}  # dict: theta: atwv val
for theta in np.arange(start_theta, end_theta, step_theta):
    print('**********************************************************************')
    print('******************************THETA = {}*****************************'.format(theta))
    print('**********************************************************************')

    train_speech_yolo.test(test_loader, model,loss.loss, config_dict, theta, args.iou_threshold, args.cuda)
    train_speech_yolo.evaluation_measures(test_loader, model, theta, config_dict, args.cuda)
    # atwv = train_speech_yolo.test_atwv(test_loader, model, config_dict, theta, wav_len=1, is_cuda=args.cuda)
    # atwv_dict[theta] = atwv
    # print('**********************************************************************')

print('====> SORTED ATWV:')
for key in sorted(atwv_dict):
    print(key, atwv_dict[key])


# checkpoint11 = torch.load('vgg11_all_train_march13_arc.pth')
# checkpoint19 = torch.load('vgg19_all_train_march15_arc.pth')


# config_dict = {'C': 6, 'B': 2, 'K': 1000}

# state11 = {'acc': checkpoint11['acc'], 
# 'wav_len': checkpoint11['wav_len'], 
# 'config_dict': config_dict, 
# 'all_conv': checkpoint11['all_conv'], 
# 'epoch': checkpoint11['epoch'], 
# 'net': checkpoint11['net'], 
# 'dropout': checkpoint11['dropout'], 
# 'arc': checkpoint11['arc']}


# state19 = {'acc': checkpoint19['acc'], 
# 'wav_len': checkpoint19['wav_len'], 
# 'config_dict': config_dict, 
# 'all_conv': checkpoint19['all_conv'], 
# 'epoch': checkpoint19['epoch'], 
# 'net': checkpoint19['net'], 
# 'dropout': checkpoint19['dropout'], 
# 'arc': checkpoint19['arc']}

# torch.save(state11, 'vgg11_all_train_march13_arc.pth')
# torch.save(state19, 'vgg19_all_train_march15_arc.pth')

