
from glob import iglob
import numpy as np
from scipy.io import wavfile
from numpy.random import seed,randint
import os,pickle,mel_features,resampy
import vggish_params as params
from audio_models import vgg
from keras import backend as K
# from vggish import VGGish


def preprocess_sound(data, sample_rate):
    """
    处理音频
    """
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

def extract_audio_feature(audio_dir,audio_model):
    """
    函数作用：提取音频特征
    对音频数据进行处理，然后送入网络提取指定层的特征，output为经过VGGish网络后输出的音频特征
    """
    sr, wav_data = wavfile.read(audio_dir)
    length = sr
    seg_num=1
    data = np.zeros((params.NUM_FRAMES, params.NUM_BANDS, 1))
    ''' by wxy'''
    # 5s segment
    if (wav_data.shape[1] > 1):
        if length > wav_data.shape[0]:
            wave_temp = np.transpose(wav_data)
            leng_1 = int(length / wav_data.shape[0] + 1)
            wave_data_temp = np.tile(wave_temp, leng_1)
            # wav_data = wave_data_temp.transpose
            wav_data = np.transpose(wave_data_temp)

        range_high = len(wav_data) - length
        seed(1)  # for consistency and replication
        random_start = randint(range_high, size=seg_num)

        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]:random_start[j] + length]
            cur_wav = cur_wav / 32768.0
            cur_spectro = preprocess_sound(cur_wav, sr)
            cur_spectro = np.expand_dims(cur_spectro, 3)
            data = cur_spectro

    get_middle_output = K.function([audio_model.layers[0].input, K.learning_phase()], [audio_model.layers[12].output])
    output = get_middle_output([data,0])[0]
    return output


def extract_audio_features(a_dir,a_model):
    """
    提取静态帧、光流、音频的特征并加上分类标签、一致性标签；
    最后return的是对三种特征所有数据的整合，及对应的两种标签
    """
    features_a = []
    labels = []
    labels_consist = []
    i = 0
    for each_v in iglob(os.path.join(a_dir, '**/**/**.wav'), recursive=True):
        # v_path=os.path.join(v_dir,each_v)
        mode = each_v.split('/')[-3]    #获取视频的标签等信息
        type = each_v.split('/')[-2]
        name = each_v.split('/')[-1][:-4]
        # label = 1 if name.split('_')[0] == 'violence' else 0     #血腥视频的标签为1,非血腥的视频标签为0
        label = 1 if name.split('_')[0] == 'Violence' else 0     #血腥视频的标签为1,非血腥的视频标签为0
        audio_feature_path = os.path.join('data_VF_feature', mode, type,'5',name+ '.npy')

        audio_path = os.path.join(a_dir, mode,type, name + '.wav')
        audio_feature = extract_audio_feature(audio_path, a_model)  # (1,128)
        np.save(audio_feature_path,audio_feature.reshape((128,1)))
        print(name)
        # out_a = {'feature': a_feature,
        #          'label': label}
        # with open(pkl_path_a, 'wb') as w:
        #     pickle.dump(out_a, w)
def make_map_result(audio_dir,audio_model,result_txt):
    """
    将检测结果记录下来,用于计算map
    """
    features_a = []
    labels = []
    labels_consist = []
    i = 0
    for each_v in iglob(os.path.join(audio_dir, '**/**/**.wav'), recursive=True):
        # v_path=os.path.join(v_dir,each_v)
        mode = each_v.split('/')[-3]    #获取视频的标签等信息
        type = each_v.split('/')[-2]
        name = each_v.split('/')[-1][:-4]
        label = 1 if name.split('_')[0] == 'violence' else 0     #血腥视频的标签为1,非血腥的视频标签为0
        # audio_feature_path = os.path.join('vsd2015_feature', mode, type,name + '.npy')

        audio_path = os.path.join(audio_dir, mode,type, name + '.wav')
        audio_feature = extract_audio_feature(audio_path, audio_model_model)  # (1,128)
        np.save(audio_feature_path,audio_feature.reshape((128,1)))
        print(name)

def main():
    cur_dir=os.path.dirname(__file__)
    # d = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # 返回当前文件所在的父目录
    audio_weight=os.path.join(cur_dir,'models','VF_5_noconsist.h5')
    audio_dir=os.path.join(cur_dir,'data_violentFlow')
    # result_txt='/home/gcn/vsd2015_audio.txt'
    audio_model = vgg(notrain=False)  # 调用声频文件中的VGG模型,模型初始化
    audio_model.load_weights(audio_weight)
    extract_audio_features(audio_dir,audio_model)
    # make_map_result(audio_dir,audio_model,result_txt)

if __name__ == '__main__':
    main()