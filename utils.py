
import soundfile
import librosa
from random import getrandbits
import numpy as np
import torch


def random_onoff():                # randomly turns on or off
    return bool(getrandbits(1))


def augment_signal(y, sr, quiet=True):
    count_changes = 0
    allow_pitch, allow_dyn, allow_noise = True, True, True
    y_mod = y
    # change pitch (w/o speed)
    if (allow_pitch) and random_onoff():
        bins_per_octave = 24  # pitch increments are quarter-steps
        pitch_pm = 4  # +/- this many quarter steps
        pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)
        if not quiet:
            print("    pitch_change = ", pitch_change)
        y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
        count_changes += 1
    # change dynamic range
    if (allow_dyn) and random_onoff():
        dyn_change = np.random.uniform(low=0.5, high=1.1)  # change amplitude
        if not quiet:
            print("    dyn_change = ", dyn_change)
        y_mod = y_mod * dyn_change
        count_changes += 1
        # add noise
    if (allow_noise) and random_onoff():
        noise_amp = 0.005 * np.random.uniform() * np.amax(y)
        if random_onoff():
            if not quiet:
                print("    gaussian noise_amp = ", noise_amp)
            y_mod += noise_amp * np.random.normal(size=len(y))
        else:
            if not quiet:
                print("    uniform noise_amp = ", noise_amp)
            y_mod += noise_amp * np.random.uniform(size=len(y))
        count_changes += 1

    # last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
    if (0 == count_changes):
        if not quiet:
            print("No changes made to signal, trying again")
        y_mod = augment_signal(y_mod, sr, quiet=quiet)

    return y_mod


def spect_loader(path, window_size=.02, window_stride=.01, window='hamming', normalize=True, max_len=101, augment=False):
    y, sr = soundfile.read(path)  # much faster than librosa!
    y_original_len = len(y)
    if augment:
        y = augment_signal(y, sr)
    if not len(y) == y_original_len:
        print('augmentation ruined the audio files length!!!')
        exit()

    try:
        n_fft = int(sr * window_size)
    except:
        print (path)
    win_length = n_fft
    hop_length = int(sr * window_stride)

    # STFT
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=window)
    spect, phase = librosa.magphase(D)

    spect = np.log1p(spect)
    real_features_len = spect.shape[1]
    # make all spects with the same dims
    if spect.shape[1] < max_len:
        pad = np.zeros((spect.shape[0], max_len - spect.shape[1]))
        spect = np.hstack((spect, pad))
    elif spect.shape[1] > max_len:
        spect = spect[:, :max_len]

    if spect.shape[0] < 160:
        pad = np.zeros((160 - spect.shape[0], spect.shape[1]))
        spect = np.vstack((spect, pad))
    elif spect.shape[0] > 160:
        spect = spect[:160, :]
    spect = np.resize(spect, (1, spect.shape[0], spect.shape[1]))
    spect = torch.FloatTensor(spect)

    # z-score normalization
    if normalize:
        mean = spect.mean()
        std = spect.std()
        if std != 0:
            spect.add_(-mean)
            spect.div_(std)

    return spect, len(y), real_features_len, sr


def calc_iou(pred, target):

    pred_start, pred_end = pred[0], pred[1]
    target_start, target_end = target[0], target[1]

    intersect_start = max(pred_start, target_start)
    intersect_end = min(pred_end, target_end)
    intersect_w = intersect_end - intersect_start

    if intersect_w < 0:  # no intersection
        intersect_w = 0.0

    pred_len = pred_end - pred_start
    target_len = target_end - target_start

    union = pred_len + target_len - intersect_w
    iou = float(intersect_w) / union
    return iou


def extract_data(out_tesor, C, B, K):

    out_coords = out_tesor[:, :, :3 * B].contiguous().view(-1, C, B, 3)
    out_xs = out_coords[:, :, :, 0].view(-1, C, B) / float(C)
    out_ws = torch.pow(out_coords[:, :, :, 1].view(-1, C, B), 2)
    out_start = (out_xs - (out_ws * 0.5))
    out_end = (out_xs + (out_ws * 0.5))
    pred_class_prob = out_tesor[:, :, 3 * B:].contiguous().view(-1, C, K)
    pred_class_prob = pred_class_prob.unsqueeze(2).repeat(1, 1, B, 1).view(-1, C, B, K)
    pred_conf = out_coords[:, :, :, 2].view(-1, C, B)
    return out_ws, out_start, out_end, pred_conf, pred_class_prob