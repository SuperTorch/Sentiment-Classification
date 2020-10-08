import os

# dir = "vsd2015_data/test"

def ListFilesToTxt(dir, file, filelabel,wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):   #filter all the wav file
                    class_id = name.split('_')[0]
                    #if class_id == '1' or class_id=='3':
                    if class_id == 'blood':
                        label = 1
                    else:
                        label = 0
                    filelabel_name= str(int(label))+"\n"
                    filename = dir + '/' + name + ',' +filelabel_name
                    file.write(filename)
                    filelabel.write(filelabel_name)
                    break
def ListFilesToTxt_vsd2015(dataset,usage, file, filelabel,wildcard, recursion):
    files = os.listdir(dataset+'/'+usage)
    for name in files:
        fullname = os.path.join(dataset+'/'+usage, name)
        if os.path.isdir(fullname):
            if name=='blood':
                label = 1
            else:
                label = 0
            video_list=os.listdir(fullname)
            for video in video_list:
                if (video.endswith(wildcard)):   #filter all the wav file
                    filelabel_name= str(int(label))+"\n"
                    filename = dataset+'/'+usage + '/' + name +'/'+video+ ',' +filelabel_name
                    file.write(filename)
                    filelabel.write(filelabel_name)
                    # if label==1:
                    #     for j in range(20):
                    #         file.write(filename)
                    #         filelabel.write(filelabel_name)

def Test(dataset,usage):

    outfile = dataset+'/'+usage+'/'+usage+"_noconsist.txt"
    wildcard = ".wav"

    outlabelfile=dataset+'/'+usage+'/'+usage+"_labelinput_noconsist.txt"
    file = open(outfile, "w")
    filelabel = open(outlabelfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)

    ListFilesToTxt_vsd2015(dataset,usage, file, filelabel, wildcard, 1)

    file.close()
    filelabel.close()

def TestFoldValidation(dataset,fold):
    files = os.listdir(dataset)
    all_vio_videos,all_non_videos=[],[]
    for name in range(fold):
        # if '.' not in name:   #simply identify whether name is a file or directory
        name=str(name+1)
        types=os.listdir(dataset+'/'+name)
        for type in types:
            videos=os.listdir(dataset+'/'+name+'/'+type)
            videos.append(dataset + '/' + name + '/' + type+'/')
            if 'non' not in type and 'Non' not in type:   #non violence or violence
                all_vio_videos.append(videos)
            else:
                all_non_videos.append(videos)
    for i in range(fold):
        train_file=open(dataset+'/'+'train'+'_'+str(i+1)+"_noconsist.txt",'w')
        test_file = open(dataset + '/' + 'test' + '_' + str(i + 1) + "_noconsist.txt", 'w')
        # label_file = open(dataset + '/' + 'train' + '_' + str(i + 1) + "_noconsist_label.txt", 'w')
        for j in range(fold):
            if i!=j:
                for k in range(len(all_non_videos[j])-1):
                    train_file.write(all_non_videos[j][-1]+all_non_videos[j][k]+','+'0'+'\n')
                    # label_file.write('0' + '\n')
                    train_file.write(all_vio_videos[j][-1]+all_vio_videos[j][k]+','+'1'+'\n')
                    # label_file.write('1' + '\n')
            else:
                for k in range(len(all_non_videos[j])-1):
                    test_file.write(all_non_videos[j][-1]+all_non_videos[j][k]+','+'0'+'\n')
                    test_file.write(all_vio_videos[j][-1]+all_vio_videos[j][k]+','+'1'+'\n')


if __name__=='__main__':
    dataset='data_violentFlow'
    usage='trainval'
    fold=5
    TestFoldValidation(dataset,fold)
    # Test(dataset,usage)