__author__ = 'YaelSegal'

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

parser = argparse.ArgumentParser(description='test  speechYolo model')
parser.add_argument('--train_data', type=str,
                    default='/data/tzya/gcommand_style/split_data/LibriSpeech_cnvt_Train_960_no_overlap',
                    help='location of the train data')
parser.add_argument('--test_data', type=str,
                    default='/data/tzya/gcommand_style/split_data/LibriSpeech_cnvt_Test_words_960',
                    help='location of the train data')
parser.add_argument('--model', type=str,
                    default='/home/mlspeech/segalya/yolo/YoloSpeech2Word/burst_closure_voicing_yolo/new_models/loss_30adam_0.0001_VGG11_1_30_s_10_b_2_c_8.pth',
                    help='the location of the trained speech yolo model')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--seed', type=int, default=1245,
                    help='random seed')
parser.add_argument('--theta_range', type=str, default='0.1_1.0_0.1', help='0.0_1.0_0.1')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--iou_threshold', type=float, default=0.5,
                    help='high iou threshold')

args = parser.parse_args()

args.cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model, acc, epoch = load_model(args.model, args.arc)
config_dict = {"C": model.c, "B": model.b, "K": model.k}
if args.cuda:
    print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model).cuda()

test_dataset = Datasets.SpeechYoloDataSet(classes_root_dir=args.train_data, this_root_dir=args.test_data,
                                          yolo_config=config_dict)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20,
                                          pin_memory=args.cuda, sampler=None)

range_split = args.theta_range.split('_')
start_theta = float(range_split[0])
end_theta = float(range_split[1])
step_theta = float(range_split[2])

# threshold = args.decision_threshold
atwv_dict = {}  # dict: theta: atwv val
for theta in np.arange(start_theta, end_theta, step_theta):
    print('**********************************************************************')
    print('******************************THETA = {}*****************************'.format(theta))
    print('**********************************************************************')
    train_speech_yolo.test(test_loader, model, config_dict, theta, args.iou_threshold, args.cuda)
    train_speech_yolo.test_acc(test_loader, model, theta, config_dict, args.cuda)
    atwv = train_speech_yolo.test_mtwv(test_loader, model, config_dict, theta, wav_len=1, is_cuda=args.cuda)
    atwv_dict[theta] = atwv
    # print('**********************************************************************')

print('====> SORTED ATWV:')
for key in sorted(atwv_dict):
    print(key, atwv_dict[key])
