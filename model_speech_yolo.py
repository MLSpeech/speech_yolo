__author__ = 'YaelSegal & TzeviyaFuchs'
# VGG model was taken from Yossi Adi
import torch.nn as nn
import torch.nn.functional as F
import torch


def _make_layers(cfg, kernel=3):
    layers = []
    in_channels = 1
    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=kernel, padding=1),
                       nn.BatchNorm2d(x),
                       nn.ReLU(inplace=True)]
            in_channels = x
    layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
    return nn.Sequential(*layers)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, class_num=30):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg[vgg_name])
        self.fc1 = nn.Linear(7680, 512)
        self.fc2 = nn.Linear(512, class_num)

    def forward(self, x):
        # out = self.features(x)
        for m in self.features.children():
            x = m(x)

        out = x
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, dim=1)


class SpeechYoloVGGNet(nn.Module):

    def __init__(self, classfication_model, c=7, b=2, k=10, dropout=0):
        super(SpeechYoloVGGNet, self).__init__()
        self.c = c
        self.b = b
        self.k = k
        self.dropout_exists = dropout
        self.dropout = nn.Dropout(p=dropout)
        self.removed = list(classfication_model.children())[:-1]

        last_input_size = 512
        self.model = nn.Sequential(*self.removed)
        self.batch_norm = nn.BatchNorm1d(last_input_size)
        self.last_layer = nn.Linear(last_input_size, self.c * (self.b * 3 + self.k))
        self.init_weight()

    def forward(self, x):
        for m in self.model.children():
            classname = m.__class__.__name__
            if 'Sequential' in classname:
                for seq_child in m.children():
                    x = seq_child(x)
            elif classname.find('Linear') != -1:
                x = x.view(x.size(0), -1)
                x = F.relu(m(x))
                if not self.dropout_exists == 0.0:
                    x = self.dropout(x)
        last_layer_output = self.last_layer(x)
        reshaped_output = last_layer_output.contiguous().view(-1, self.c, self.b * 3 + self.k)
        pred_coords = reshaped_output[:, :, :3 * self.b].contiguous().view(-1, self.c, self.b,
                                                                           3)  # get all the x,w values for all the boxes
        target_xs = torch.sigmoid(pred_coords[:, :, :, 0].view(-1, self.c, self.b))
        target_ws = torch.sigmoid(pred_coords[:, :, :, 1].view(-1, self.c, self.b))
        target_conf = torch.sigmoid(pred_coords[:, :, :, 2].view(-1, self.c, self.b))
        target_class_prob = F.softmax(reshaped_output[:, :, 3 * self.b:].contiguous().view(-1, self.c, self.k), 2)
        final_output = torch.cat((target_xs, target_ws, target_conf, target_class_prob), 2)
        return final_output

    def init_weight(self):
        torch.nn.init.xavier_normal_(self.last_layer.weight.data)

    def init_mult_weights(self):
        torch.nn.init.xavier_normal_(self.last_layer[0].weight.data)
        torch.nn.init.xavier_normal_(self.last_layer[2].weight.data)

        # [1] is batchnorm
        self.last_layer[1].weight.data.normal_(1.0, 0.02)
        self.last_layer[1].bias.data.normal_(1.0, 0.02)

    @staticmethod
    def init_pre_model_weights(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1 or classname.find('Linear') != -1:
            nn.init.xavier_normal_(m.weight)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.normal_(1.0, 0.02)


def load_model(save_dir):
    checkpoint = torch.load(save_dir, map_location=lambda storage, loc: storage)
    config_dict = checkpoint['config_dict']
    arc_type = checkpoint['arc']

    if arc_type.startswith('VGG'):
        model_ = VGG(arc_type)
        speech_net = SpeechYoloVGGNet(model_, config_dict["C"], config_dict["B"], config_dict["K"])
        speech_net.load_state_dict(checkpoint['net'])

    else:
        raise Exception("No such architecture")

    return speech_net, checkpoint['acc'], checkpoint['epoch']


def create_speech_model(pretrained_model, arc, config_dict, dropout):
    if pretrained_model:
        if arc.startswith('VGG'):
            checkpoint = torch.load(pretrained_model, map_location=lambda storage,
                                                                          loc: storage)  # will forcefully remap everything onto CPU
            class_num = checkpoint['class_num']
            model_ = VGG(arc, class_num=class_num)
            model_.load_state_dict(checkpoint['net'])
            speech_net = SpeechYoloVGGNet(model_, config_dict["C"], config_dict["B"], config_dict["K"], dropout=dropout)
        else:
            raise Exception("No such architecture")
    else:
        if arc.startswith('VGG'):
            model_ = VGG(arc)
            speech_net = SpeechYoloVGGNet(model_, config_dict["C"], config_dict["B"], config_dict["K"], dropout=dropout)
            speech_net.model.apply(speech_net.init_pre_model_weights)
        else:
            raise Exception("No such architecture")
    return speech_net
