import os
import numpy as np
import torch.nn.functional as F
import torch
import subprocess

def make_txt(result_txt):
    result_dir=os.path.dirname(__file__)
               # +'/fusion/fuison_result.txt'
    # result_txt='/home/gcn/caffe_lstm_result.txt'

    # result_txt='lstm/lstm_result_fuse.txt'
    # result_txt=result_dir+'/lstm_fuse/lstm_fuse_aug_result.txt'
    result_file=open(result_txt,'r')
    vio_qrels='/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_vio_qrels.test'
    vio_results='/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_vio_results.test'
    non_qrels='/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_non_qrels.test'
    non_results='/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_non_results.test'
    # test_file=open('/home/gcn/gcn/videoClassification/trec_eval.9.0/test/results.test','r')
    # lslsl=test_file.readline()
    vio_qrelsFile=open(vio_qrels,'w')
    vio_resultsFile=open(vio_results,'w')
    non_qrelsFile=open(non_qrels,'w')
    non_resultsFile=open(non_results,'w')
    result_lines=result_file.readlines()
    for i in range(len(result_lines)):
        result= result_lines[i]
        non_score, vio_score, label = float(result.split(' ')[0]), float(result.split(' ')[1].strip()), int(
            result.split(' ')[2].strip())
        softmax_scores=F.softmax(torch.Tensor([non_score, vio_score]), dim=0).detach()
        non_score=np.array(softmax_scores)[0]
        vio_score=np.array(softmax_scores)[1]
        torch.max(softmax_scores, dim=0)
        vio_qrelsFile.write('301 0 '+str(i+1).zfill(6)+' '+str(label)+'\n')
        vio_resultsFile.write('301\tQ0\t'+ str(i+1).zfill(6)+'\t'+ str(i+1).zfill(6)+'\t'+str(vio_score)+'\tSTANDARD\n')
        non_label=1 if label==0 else 0
        non_qrelsFile.write('301 0 ' + str(i+1).zfill(6) + ' ' + str(non_label) + '\n')
        non_resultsFile.write('301\tQ0\t' + str(i+1).zfill(6) + '\t'+ str(i+1).zfill(6)+'\t' + str(non_score) + '\tSTANDARD\n')
    vio_qrelsFile.close()
    vio_resultsFile.close()
    non_qrelsFile.close()
    non_resultsFile.close()

    # (status, output) = subprocess.getstatusoutput('pwd')
    (status, output) = subprocess.getstatusoutput(
        './../../../../../gcn/videoClassification/trec_eval.9.0/trec_eval -m map '
        '/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_vio_qrels.test '
        '/home/gcn/gcn/videoClassification/trec_eval.9.0/test_txt/c_vio_results.test')    #执行sh文件,调用p3d用来提取特征
    # print(status)
    print('\033{}'.format(output))
# if __name__ == '__main__':
#     result_txt = 'vsd2015_data/test/vsd2015_audio_test_score.txt'
#     # sh_txt='/home/gcn/gcn/videoClassification/trec_eval.9.0/sh_file/compute_map.sh'
#     make_txt(result_txt)
