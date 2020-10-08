from __future__ import division

import sys


#sys.path.append('C:/Users/wxy/Anaconda3/Lib/site-packages/h5py')
#sys.path.append('/home/hudi/anaconda2/lib/python2.7/site-packages/Keras-2.0.6-py2.7.egg')

import numpy as np
from numpy.random import seed, randint
from scipy.io import wavfile
from sklearn import svm
import linecache

from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from vggish import VGGish

import vggish_params as params
import mel_features
import resampy
from keras.utils.vis_utils import plot_model

seg_num = 1
def preprocess_sound(data, sample_rate):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.
  """
  # Convert to mono.

  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=params.SAMPLE_RATE,
      log_offset=params.LOG_OFFSET,
      window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=params.NUM_MEL_BINS,
      lower_edge_hertz=params.MEL_MIN_HZ,
      upper_edge_hertz=params.MEL_MAX_HZ)

  # Frame features into examples.
  features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS
  example_window_length = int(round(
      params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(
      params.EXAMPLE_HOP_SECONDS * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)
  return log_mel_examples

def loading_data(files, sound_extractor):

    lines = linecache.getlines(files)
    sample_num = len(lines)

    seg_len = 1  # 5s
    data = np.zeros((seg_num * sample_num, params.NUM_FRAMES, params.NUM_BANDS, 1))
    label = np.zeros((seg_num * sample_num,))

    for i in range(len(lines)):

        sound_file = lines[i].split(',')[0]
        label_id = lines[i].split(',')[1].split()[0]
        sr, wav_data = wavfile.read(sound_file)

        length = sr * seg_len

        range_high = len(wav_data) - length
        seed(1)  # for consistency and replication
        random_start = randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i * seg_num + j, :, :, :] = cur_spectro
            #label[i * seg_num + j] = lines[i][-2]
            label[i * seg_num + j] = lines[i].split(',')[1].split()[0]

    data = sound_extractor.predict(data)

    return data, label

def loading_data_modify(files, sound_extractor):

    lines = linecache.getlines(files)
    sample_num = len(lines)

    seg_len = 1  # 5s
    data = np.zeros((seg_num * sample_num, params.NUM_FRAMES, params.NUM_BANDS, 1))
    label = np.zeros((seg_num * sample_num,))

    for i in range(len(lines)):

        sound_file = lines[i].split(',')[0]
        label_id = lines[i].split(',')[1].split()[0]
        sr, wav_data = wavfile.read(sound_file)

        length = sr * seg_len

        ''' by wxy'''
        # 5s segment
        if( wav_data.shape[1]>1):
            if length > wav_data.shape[0]:
                wave_temp=wav_data.T
                leng_1 = int(length / wav_data.shape[0] + 1)
                wave_data_temp = np.tile(wave_temp, leng_1)
                wav_data = wave_data_temp.T


      #  wav_mono_data=raw_audio.reshape(raw_audio.shape[0],1)
        #raw_audio = raw_audio[:length]

        range_high = len(wav_data) - length
        seed(1)  # for consistency and replication
        random_start = randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data[i * seg_num + j, :, :, :] = cur_spectro
            #label[i * seg_num + j] = lines[i][-2]
            label[i * seg_num + j] = lines[i].split(',')[1].split()[0]

    data = sound_extractor.predict(data)

    return data, label

def train_finetuen():
    sound_model = VGGish(include_top=False, load_weights=True)


    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    print("loading training data...")
    # training_file = 'data/train/train_all_list.txt'
    training_file = 'data/train/train_all_list.txt'

    training_data, training_label = loading_data_modify(training_file, sound_extractor)

    # load testing data
    print("loading testing data...")
    # testing_file = 'data/test/test_all_list.txt'
    testing_file = 'data/test/test_all_list.txt'
    testing_data, testing_label = loading_data(testing_file, sound_extractor)


    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(training_data, training_label.ravel())
    p_vals = clf.decision_function(testing_data)

    testing_lines = linecache.getlines(testing_file)
    test_count_seg = len(testing_label)
    test_count = len(testing_lines)

    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:test_count_seg:seg_num]
    # gt = testing_label[0:6000:60]
    # gt = testing_label
    p_vals = np.asarray(p_vals)

    outfile = "results/test_result.txt"
    file = open(outfile, "a+")

    thresh_value = 0
    for ii in range(test_count):
        scores = np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)
        # ind = np.argmax(scores)

        # scores = p_vals[ii]
        if scores > thresh_value:
            ind = 1
        else:
            ind = 0;

        pred_labels[ii] = ind
    # pred_labels=p_vals
    scores = gt == pred_labels
    score = np.mean(scores)
    file.write('acc:%.3f,thresh:%.2f load_weights=True\n' % (score, thresh_value))
    print("accuracy: %f" % score)
    file.close()

def svm_result():
    sound_model = VGGish(include_top=True, load_weights=True)
    print(sound_model.summary())
    #plot_model(sound_model, to_file='models/model_structure.png', show_shapes=True)

  #<?,12,8,512>
    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    #output_layer <?,512>
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    print("loading training data...")
    # training_file = 'data/train/train_all_list.txt'
    training_file = 'data/train/train_all_list.txt'

    training_data, training_label = loading_data_modify(training_file, sound_extractor)
   #<trian_data:<?,512>
    # load testing data
    print("loading testing data...")
    # testing_file = 'data/test/test_all_list.txt'
    testing_file = 'data/test/test_all_list.txt'
    testing_data, testing_label = loading_data(testing_file, sound_extractor)

    clf = svm.LinearSVC(C=0.01, dual=False)
    clf.fit(training_data, training_label.ravel())
    p_vals = clf.decision_function(testing_data)

    testing_lines = linecache.getlines(testing_file)
    test_count_seg = len(testing_label)
    test_count = len(testing_lines)

    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:test_count_seg:seg_num]
    # gt = testing_label[0:6000:60]
    # gt = testing_label
    p_vals = np.asarray(p_vals)

    outfile = "results/test_result.txt"
    file = open(outfile, "a+")

    thresh_value = 0
    for ii in range(test_count):
        scores = np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)
        # ind = np.argmax(scores)

        # scores = p_vals[ii]
        if scores > thresh_value:
            ind = 1
        else:
            ind = 0;

        pred_labels[ii] = ind
    # pred_labels=p_vals
    scores = gt == pred_labels
    score = np.mean(scores)
    file.write('svm methods: acc:%.3f,thresh:%.2f load_weights=True\n' % (score, thresh_value))
    print("accuracy: %f" % score)
    file.close()

if __name__ == '__main__':
    svm_result()



