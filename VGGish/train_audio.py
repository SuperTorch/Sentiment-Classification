#coding=utf-8
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
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense,Activation
from keras import applications
from keras.utils.vis_utils import plot_model
from keras import regularizers

from keras.layers.convolutional import Conv1D, MaxPooling1D,AveragePooling1D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint

import vggish_params as params
import mel_features
import resampy
from vggish import VGGish
from random import shuffle
from make_trec_eval_txt import make_txt


seg_num = 1
num_classes =2
epochs = 10
#epochs=300

batch_size = 4

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
'''
load audio file, extract their feature, feature dimensions is 512
'''
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

'''
load audio file, extract their feature, feature dimensions is 512
the difference with loading_data is the following:
   if the length of audio file is less than setting length(for example seg_len=1s), it will be extended to the setting length.
'''
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


'''
load training raw audio file
convert audio into 96x64 mfcc
'''
def loading_data_audio(files):

    lines = linecache.getlines(files)
    shuffle(lines)
    sample_num = len(lines)

    seg_len = 1  # 5s
    data = np.zeros((seg_num * sample_num, params.NUM_FRAMES, params.NUM_BANDS, 1))
    label = np.zeros((seg_num * sample_num,))
    video_name=[]

    for i in range(len(lines)):
        video_name.append(lines[i].split(',')[0].split('/')[-1])

        sound_file = lines[i].split(',')[0]
        label_id = lines[i].split(',')[1].split()[0]
        sr, wav_data = wavfile.read(sound_file)   #读取声音的wav文件,第一个返回值为声音采样率,第二个返回值为声音文件

        length = sr * seg_len

        ''' by wxy'''
        # 5s segment
        if( wav_data.shape[1]>1):
            if length > wav_data.shape[0]:   #如果音频数据长度较小,则对音频数据进行堆叠,以达到要求
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
            cur_spectro = preprocess_sound(cur_wav, sr)   #将声音文件处理成VGG模型可以输入的类型
            cur_spectro = np.expand_dims(cur_spectro, 3)   #用于对数据的维度进行扩展,此处是由3维数据变换到4维数据
            data[i * seg_num + j, :, :, :] = cur_spectro    #将所有的声频文件聚合到一起
            #label[i * seg_num + j] = lines[i][-2]
            label[i * seg_num + j] = lines[i].split(',')[1].split()[0]   #将声频文件的标签进行聚合

    return data, label,video_name
'''
extract the feature from audio file. the feature is 96x64 mfcc
'''
def extract_feature(traing_path,testing_path):
    ''' extract the feature of audio files'''
    sound_model = VGGish(include_top=False, load_weights=False)
    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    print("loading training data...")
    # training_file = 'data/train/train_all_list.txt'
    training_file = traing_path#'data/train/train_all_list.txt'

    training_data, training_label = loading_data_modify(training_file, sound_extractor)
    training_data = np.expand_dims(training_data, 2)

    train_feature_file='features/train_feature.npy'
    np.save(train_feature_file, training_data)
    train_label_file='features/train_label.npy'
    np.save(train_label_file, training_label)
    ''''''''
    # load testing data
    print("loading testing data...")
    # testing_file = 'data/test/test_all_list.txt'
    test_file = testing_path#'data/test/test_all_list.txt'
    testing_data, testing_label = loading_data_modify(test_file, sound_extractor)
    testing_data = np.expand_dims(testing_data, 2)

    test_feature_file = 'features/test_feature.npy'
    np.save(test_feature_file, testing_data)
    test_label_file = 'features/test_label.npy'
    np.save(test_label_file, testing_label)
'''
 the extracted feature ahead is input, train the fc classification network
'''
def train_fc_audio(testing_file):

    '''load testing feature'''
    train_feature_path = 'features/train_feature.npy'
    training_data = np.load(open(train_feature_path, 'rb'))
    train_label_path = 'features/train_label.npy'
    training_label = np.load(open(train_label_path, 'rb'))

    '''load testing feature'''
    test_feature_path = 'features/test_feature.npy'
    testing_data = np.load(open(test_feature_path, 'rb'))
    test_label_path = 'features/test_label.npy'
    testing_label = np.load(open(test_label_path, 'rb'))


    top_model = Sequential()
    top_model_weights_path = 'models/bottleneck_fc_pool_model.h5'

    top_model.add(Flatten(input_shape=training_data.shape[1:]))
    top_model.add(Dense(128, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    print(top_model.summary())
    plot_model(top_model, to_file='models/model_fc_pooling_structure.png', show_shapes=True)

    #首先使用的
    # top_model.compile(optimizer='rmsprop',
    #               loss='binary_crossentropy', metrics=['accuracy'])
    top_model.compile(optimizer='rmsprop',
                      loss='binary_crossentropy', metrics=[])


    history = top_model.fit(training_data, training_label,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(testing_data, testing_label))



    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("results/acc_fc_pooling.png")
    #plt.show()

    top_model.save_weights(top_model_weights_path)

    testing_lines = linecache.getlines(testing_file)
    test_count_seg = len(testing_label)
    test_count = len(testing_lines)

    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:test_count_seg:seg_num]

    p_vals = top_model.predict(testing_data,batch_size=batch_size)
    p_vals = np.asarray(p_vals)

    outfile = "results/test_fc_result.txt"
    file = open(outfile, "a+")

    thresh_value = 0.5
    for ii in range(test_count):
        scores = np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)

        if scores > thresh_value:
            ind = 1
        else:
            ind = 0

        pred_labels[ii] = ind
    # pred_labels=p_vals
    scores = gt == pred_labels
    score = np.mean(scores)
    file.write('acc:%.3f,thresh:%.2f load_weights=True\n' % (score, thresh_value))
    # print("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f" % ( history.history['acc'][epochs - 1], history.history['loss'][epochs - 1], history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))
    file.write("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f\n" % (
        history.history['acc'][epochs - 1], history.history['loss'][epochs - 1],
        history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))

    print("accuracy: %f" % score)
    file.close()


'''set the whole netowrk'''
def build_model(notrain):

    # create the base pre-trained model
    sound_model = VGGish(include_top=False, load_weights=True)    #加载VGG网络结构,病利用预训练模型进行初始化
    x=sound_model.output     #获得模型的输出结构

    # let's add a fully-connected layer
    x = Dense(128, activation='relu')(x)   #在标准VGG模型的输出上计算准确率
    x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 2classes ，siogmoid fucntion

    predictions = Dense(1, activation='sigmoid',name='finetune_new')(x)

    # this is the model we will train
    model = Model(inputs=sound_model.input, outputs=predictions)
    #model.load_weights('models/addnew_consist2019.h5', by_name=True)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers


    if notrain:
       for layer in sound_model.layers[:10]: # for layer in sound_model.layers:
            layer.trainable = False   #前10层不训练

    print(model.summary())
    #plot_model(model, to_file='models/model_fc_pooling_structure.png', show_shapes=True)
    return model

'''
build the network, using  audioset weight to fine tune the network, fc layer is set to be trainable
'''
def train_finetune_result():
    # create the base pre-trained model
    model = build_model(notrain=True)    #定义模型文件,用预训练模型进行初始化

    # load training data
    print("loading training data...")
    training_file = 'data_violentFlow/train_1_noconsist.txt'
    # training_file = 'data_wg/train/train_noconsist.txt'
    # training_file = 'vsd2015_data/trainval/trainval_train_noconsist.txt'
    # training_file = 'vsd2015_data/trainval/trainval_modify.txt'
    training_data, training_label,train_video_name = loading_data_audio(training_file)    #根据训练文件标示,提取声音文件

    # load testing data
    print("loading testing data...")
    # testing_file = 'data/test/test_all_list.txt'
    # testing_file = 'data_wg/val/val_noconsist.txt'
    testing_file = 'data_violentFlow/test_1_noconsist.txt'
    # testing_file = 'vsd2015_data/test/test_noconsist.txt'
    testing_data, testing_label,test_video_name = loading_data_audio(testing_file)

    model.compile(optimizer='rmsprop',      #模型编译,参数为最优化函数和loss函数,同时给定准确率计算方式
                      loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(training_data, training_label,     #给定训练数据和标签及训练参数,训练网络模型,同时记录模型训练过程
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(testing_data, testing_label))

    # summarize history for accuracy
    # plt.plot(history.history['acc'])     #画出训练及测试的loss曲线
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.savefig("results/vsd2015_noconsist.png")
    # plt.show()
    model_weights_path="models/VF_1_noconsist.h5"     #保存训练后的模型
    model.save_weights(model_weights_path)

    testing_lines = linecache.getlines(testing_file)
    test_count_seg = len(testing_label)
    test_count = len(testing_lines)

    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:test_count_seg:seg_num]

    p_vals = model.predict(testing_data, batch_size=batch_size)
    p_vals = np.asarray(p_vals)

    outfile = "results/wg_test_finetune_result.txt"
    file = open(outfile, "a+")


    # sound_result='compute_map.txt'
    # sound_file=open(sound_result,'w')
    # score_txt='vsd2015_data/test/vsd2015_audio_test_score.txt'
    score_txt='results/VF/test_1_score.txt'
    score_file=open(score_txt,'w')
    thresh_value = 0.5
    tp,tn,fp,fn=0,0,0,0
    for ii in range(test_count):     #计算准确率
        np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)
        vio_scores = np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)
        non_scores = 1 - vio_scores
        vio_score2write = str(float(vio_scores))[:10] if len(str(float(vio_scores))) >= 10 else str(
            float(vio_scores)).zfill(10)
        non_score2write = str(float(non_scores))[:10] if len(str(float(non_scores))) >= 10 else str(
            float(non_scores)).zfill(10)
        score_file.write(non_score2write + ' ' + vio_score2write + ' ' + str(int(gt[ii])) + '\n')
        if vio_scores > thresh_value:
            ind = 1
        else:
            ind = 0

        pred_labels[ii] = ind
        if ind ==1:
            if gt[ii]==1:
                tp+=1
            else:
                fp+=1
        else:
            if gt[ii]==1:
                fn+=1
            else:
                tn+=1
    # pred_labels=p_vals
    score_file.close()
    make_txt(score_txt)
    scores = gt == pred_labels
    score = np.mean(scores)
    # file.write('acc:%.3f,thresh:%.2f\n' % (score, thresh_value))
    # # print("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f" % ( history.history['acc'][epochs - 1], history.history['loss'][epochs - 1], history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))
    # file.write("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f\n" % (
    #     history.history['acc'][epochs - 1], history.history['loss'][epochs - 1],
    #     history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))

    print("accuracy: %f" % score)
    print('tp is {}, tn is {}, fp is {}, fn is {}'.format(tp,tn,fp,fn))
    print('precision is {}, recall is {}, F1 is {}'.format(float(tp) / (tp + fp), float(tp) / (tp + fn),
                                                           float(2 * tp) / (2 * tp + fp + fn)))
    # file.close()

def svm_result():
    sound_model = VGGish(include_top=True, load_weights=True)
    print(sound_model.summary())
    plot_model(sound_model, to_file='models/model_structure.png', show_shapes=True)

  #<?,12,8,512>
    x = sound_model.get_layer(name="conv4/conv4_2").output
    output_layer = GlobalAveragePooling2D()(x)
    #output_layer <?,512>
    sound_extractor = Model(input=sound_model.input, output=output_layer)

    # load training data
    print("loading training data...")
    # training_file = 'data/train/train_ all_list.txt'
    # training_file = 'data/train/train_short_list.txt'
    training_file = 'vsd2015_data/trainval/trainval_noconsist.txt'

    training_data, training_label = loading_data_modify(training_file, sound_extractor)
   #<trian_data:<?,512>
    # load testing data
    print("loading testing data...")
    # testing_file = 'data/test/test_all_list.txt'
    # testing_file = 'data/test/test_all_list.txt'
    testing_file = 'vsd2015_data/test/test_noconsist.txt'
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

def fc_result():
    training_file = 'vsd2015_data/trainval/trainval_noconsist.txt'
    test_file = 'vsd2015_data/test/test_noconsist.txt'
    # extract_feature(training_file, test_file)
    train_fc_audio(test_file)

def test_result():
    model = build_model(notrain=False)  # 定义模型文件,用预训练模型进行初始化
    print("loading testing data...")
    # testing_file = 'data_violentFlow/test_1_noconsist.txt'
    testing_file = 'data_violentFlow/test_2_noconsist.txt'
    # testing_file = 'vsd2015_data/test/test_noconsist.txt'
    # testing_file = 'data_wg/test/test_noconsist.txt'
    # testing_file = 'vsd2015_data/trainval/trainval_test_noconsist.txt'
    testing_data, testing_label,video_name = loading_data_audio(testing_file)
    model_weights_path="models/VF_2_noconsist.h5"
    # model_weights_path = "models/vsd2015_noconsist_new.h5"  # 保存训练后的模型
    # model_weights_path="models/wg_audio_model.h5"     #保存训练后的模型
    # model_weights_path="models/wg_noconsist.h5"     #保存训练后的模型
    # model.load_weights(model_weights_path)
    model.load_weights(model_weights_path)
    # score_txt='vsd2015_data/test/vsd2015_audio_test_score.txt'
    # score_txt='data_wg/test/wg_audio_test_score.txt'
    score_txt='/home/gcn/VF_audio_run2_test.txt'
    score_file=open(score_txt,'w')

    testing_lines = linecache.getlines(testing_file)
    test_count_seg = len(testing_label)
    test_count = len(testing_lines)

    pred_labels = np.zeros((test_count,))
    gt = testing_label[0:test_count_seg:seg_num]

    p_vals = model.predict(testing_data, batch_size=batch_size)
    p_vals = np.asarray(p_vals)
    # print(str(p_vals))

    outfile = "results/trainval_finetune_result.txt"
    file = open(outfile, "a+")

    # sound_result = '/home/gcn/vggish_audio_result.txt'
    # sound_file = open(sound_result, 'w')
    thresh_value = 0.5
    tp, tn, fp, fn = 0, 0, 0, 0

    for ii in range(test_count):  # 计算准确率
        vio_scores = np.mean(p_vals[ii * seg_num:(ii + 1) * seg_num:1], axis=0)
        non_scores = 1 - vio_scores
        vio_score2write = str(float(vio_scores))[:10] if len(str(float(vio_scores))) >= 10 else str(
            float(vio_scores)).zfill(10)
        non_score2write = str(float(non_scores))[:10] if len(str(float(non_scores))) >= 10 else str(
            float(non_scores)).zfill(10)
        score_file.write(video_name[ii]+' '+non_score2write + ' ' + vio_score2write + ' ' + str(int(gt[ii])) + '\n')
        # score_file.write(non_score2write + ' ' + vio_score2write + ' ' + str(int(gt[ii])) + '\n')
        if vio_scores > thresh_value:
            ind = 1
        else:
            ind = 0

        pred_labels[ii] = ind
        if ind == 1:
            if gt[ii] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if gt[ii] == 1:
                fn += 1
            else:
                tn += 1
    # pred_labels=p_vals
    # sound_file.close
    score_file.close()
    # make_txt(score_txt)
    scores = gt == pred_labels
    score = np.mean(scores)
    file.write('acc:%.3f,thresh:%.2f\n' % (score, thresh_value))
    # print("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f" % ( history.history['acc'][epochs - 1], history.history['loss'][epochs - 1], history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))
    # file.write("train_acc:%.3f,train_loss:%.3f val_acc:%.3f,val_loss:%.3f\n" % (
    #     history.history['acc'][epochs - 1], history.history['loss'][epochs - 1],
    #     history.history['val_acc'][epochs - 1], history.history['val_loss'][epochs - 1]))

    print("accuracy: %f" % score)
    print('tp is {}, tn is {}, fp is {}, fn is {}'.format(tp, tn, fp, fn))
    print('precision is {}, recall is {}, F1 is {}'.format(float(tp) / (tp + fp), float(tp) / (tp + fn),
                                                           float(2 * tp) / (2 * tp + fp + fn)))
    file.close()

if __name__ == '__main__':
    #svm_result()
    #fc_result()
    # train_finetune_result()
    test_result()


