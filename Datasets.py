# Yossi Adi wrote ClassificationLoader  (GCommandLoader with few changes)
# Yosi Shrem wrote some of ImbalancedDatasetSampler

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
import os
import torch
import math
import utils

import pdb

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if (os.path.isdir(os.path.join(dir, d)))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


AUDIO_EXTENSIONS = [
    '.wav', '.WAV',
]


def is_audio_file(filename):
    return any(filename.endswith(extension) for extension in AUDIO_EXTENSIONS)


class ClassificationLoader(Dataset):
    """

    A  data set loader where the wavs are arranged in this way:
        root/one/xxx.wav
        root/one/xxy.wav
        root/one/xxz.wav
        root/head/123.wav
        root/head/nsdf3.wav
        root/head/asd932_.wav
    Args:
        root (string): Root directory path.
        window_size: window size for the stft, default value is .02
        window_stride: window stride for the stft, default value is .01
        window_type: typye of window to extract the stft, default value is 'hamming'
        normalize: boolean, whether or not to normalize the spect to have zero mean and one std
        max_len: the maximum length of frames to use
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        spects (list): List of (spects path, class_index) tuples
        STFT parameter: window_size, window_stride, window_type, normalize
    """

    def __init__(self, root, window_size=.02,
                 window_stride=.01, window_type='hamming', normalize=True, max_len=101):
        classes, class_to_idx = find_classes(root )

        spects = []
        dir = os.path.expanduser(root)
        count = 0
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            # if count > 20:
            #     break
            for root, _, fnames in sorted(os.walk(d)):
                # if count > 20:
                #     break
                for fname in sorted(fnames):
                    # if count > 20:
                    #     break
                    if is_audio_file(fname):
                        path = os.path.join(root, fname)
                        label = os.path.join(root, fname.replace(".wav", ".wrd"))
                        item = (path, class_to_idx[target], label)
                        spects.append(item)
                        count += 1
        if len(spects) == 0:
            raise (RuntimeError("Found 0 sound files in subfolders of: " + root + "Supported audio file extensions are: " + ",".join(AUDIO_EXTENSIONS)))

        self.root = root
        self.type = type
        self.spects = spects
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.loader = utils.spect_loader
        self.window_size = window_size
        self.window_stride = window_stride
        self.window_type = window_type
        self.normalize = normalize
        self.max_len = max_len

    def __getitem__(self, index):
        self.class__ = """
        Args:
            index (int): Index
        Returns:
            tuple: (spect, target) where target is class_index of the target class.
        """
        path, target, label_path = self.spects[index]
        spect, _, _, _ = self.loader(path, self.window_size, self.window_stride, self.window_type, self.normalize, self.max_len)

        # return features, target
        return spect, target

    def __len__(self):
        return len(self.spects)


class SpeechYoloDataSet(Dataset):
    def __init__(self, classes_root_dir, this_root_dir, yolo_config, words_list_file = None, augment=False):
        """
        :param root_dir:
        :param yolo_config: dictionary that contain the require data for yolo (C, B, K)
        """
        self.augment = augment
        self.root_dir = this_root_dir
        self.C = yolo_config["C"]
        self.B = yolo_config["B"]
        self.K = yolo_config["K"]

        if not words_list_file:

            classes, class_to_idx = find_classes(classes_root_dir)

        else:


            index2word = {}
            classes = np.loadtxt(words_list_file, dtype=str)
            if classes.size ==1:
                classes = np.array([words_list])
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}


        self.class_to_idx = class_to_idx
        self.classes = classes
        spects = []
        count = 0
        dir = os.path.expanduser(this_root_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            if target not in classes:
                continue
            # if count > 1000:
            #     break
            for root, _, fnames in sorted(os.walk(d)):
                # if count > 1000:
                #     break
                for fname in sorted(fnames):
                    # if count > 1000:
                    #     break
                    count += 1
                    if is_audio_file(fname):
                        path = os.path.join(root, fname)
                        x = os.path.getsize(path)
                        if x < 1000:
                            print (path)
                            continue
                        label = os.path.join(root, fname.replace(".wav", ".wrd"))
                        tclass = self.class_to_idx[target]
                        item = (path, label, tclass)
                        spects.append(item)
        self.data = spects

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        """
        features_path = self.data[idx][0]
        add_augment = False
        if self.augment:
            add_augment = utils.random_onoff()
        features, dot_len, real_features_len, sr = utils.spect_loader(features_path, max_len=101, augment=add_augment)
        target_path = self.data[idx][1]
        target = open(target_path, "r").readlines()

        _, num_features, features_wav_len = features.shape #(1, 160, 101)

        '''
        #   the labels file, is file with few lines,
        #  each line represents one item in the wav file and contains : start (in sr), end (in sr), class.
        # yolo needs more details
        # x - represent the center of the box relative to the bounds of the grid cell.
        # w - predicted relative to the whole wav.
        # iou - the confidence prediction represents the IOU between the predicted box and any ground truth box

        '''

        divide = sr/features_wav_len  # 16000/101 = 158.41, each feature in x contains 158.41 samples from the original wav file
        width_cell = 1.0 * features_wav_len / self.C  # width per cell  101/6 = 20.16
        line_yolo_data = []  # index, relative x, w, class

        for line_str in target:
            line = line_str.replace("\t", "_").split(" ")
            feature_start = math.floor(float(line[0])/divide)
            feature_end = math.floor(float(line[1])/divide)
            object_width = (feature_end - feature_start)
            center_x = feature_start + object_width / 2.0  # left_x + width/ 2

            cell_index = int(center_x / width_cell)  # rescale the center x to cell size
            object_norm_x = float(center_x) / width_cell - int(center_x / width_cell)

            object_norm_w = object_width/features_wav_len

            class_label = line[2]
            object_class = self.class_to_idx[class_label]
            line_yolo_data.append([cell_index, object_norm_x, object_norm_w, object_class])

        kwspotting_target = torch.ones([self.K]) * (-1)
        target = torch.zeros([self.C, (self.B*3 + self.K +1)], dtype=torch.float32)  # the last place if for noobject/object
        for yolo_item in line_yolo_data:
            index = yolo_item[0]
            x = yolo_item[1]
            w = math.sqrt(yolo_item[2])
            obj_class = yolo_item[3]
            target[index, self.B*3 + obj_class] = 1  # one hot vector
            target[index, -1] = 1  # there is object in this grid cell
            for box in range(0, self.B):
                target[index, box*3 + 2] = 1  # IOU
                target[index, box * 3] = x  # x
                target[index, box * 3 + 1] = w  # w
            kwspotting_target[obj_class] = 1

        return features, target, features_path, kwspotting_target

    def get_filename_by_index(self, idx):
        return self.data[idx][0]

    def get_class(self, idx):
        item = self.data[idx]
        return item[2]


class ImbalancedDatasetSampler(Sampler):

    def __init__(self, dataset, indices=None):
        self.dataset = dataset
        if indices != None:
            self.indices = indices
        else:
            self.indices = list(range(len(dataset)))

        self.num_samples = len(self.indices)
        labels_list = []
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(idx)
            labels_list.append(label)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[labels_list[idx]] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, idx):
        return self.dataset.get_class(idx)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


