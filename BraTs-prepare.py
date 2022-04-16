import glob
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

t1 = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*t1.nii.gz')
t2 = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*t2.nii.gz')
flair = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*flair.nii.gz')
t1ce = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*t1ce.nii.gz')
seg = glob.glob('./MICCAI_BraTS_2019_Data_Training/*/*/*seg.nii.gz')

def read_img(img_path):
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))

def make_data(data1, data2):
  """
  Make input data as h5 file format
  Depending on 'is_train' (flag value), savepath would be changed.
  """
  savepath = os.path.join('.', os.path.join('Train_data', 'train_T2flair_size80.h5'))
  #if not os.path.exists(os.path.join('.',os.path.join('checkpoint','Train_data'))):
   # os.makedirs(os.path.join('.',os.path.join('checkpoint','Train_data')))

  with h5py.File(savepath, 'w') as hf:
    hf.create_dataset('data1', data=data1)
    hf.create_dataset('data2', data=data2)

def normalize_2047(image):
    image = image.astype(dtype=np.float32)
    image = image/2047
    image[image > 1.0] = 1.0
    return image

sub_input1_sequence = []
sub_input2_sequence = []

count = 0
image_size = 80

for i in range(len(t2)):
    img1 = (read_img(t2[i])).astype(np.float32)
    img2 = (read_img(flair[i])).astype(np.float32)
    print(i)
    input_1 = img1
    input_2 = img2
    '''
    ma = np.max(np.max(np.max(input_)))
    mi = np.min(np.min(np.min(input_)))
    input_ = (input_ - mi) / (ma - mi)  # [0,1]%%%%%%%%%%%%%%%%%
    '''
    #**********   normalization
    input_1 = normalize_2047(input_1)
    input_2 = normalize_2047(input_2)

    d, h, w = input_1.shape
    for z in range(20, d - 20 - image_size + 1, 30):
        for x in range(20, h - 20 - image_size + 1, 50):
            for y in range(20, w - 20 - image_size + 1, 50):
                sub_input1 = input_1[z:z + image_size, x:x + image_size, y:y + image_size]
                sub_input1 = sub_input1.reshape([image_size, image_size, image_size, 1])
                sub_input1_sequence.append(sub_input1)

                sub_input2 = input_2[z:z + image_size, x:x + image_size, y:y + image_size]
                sub_input2 = sub_input2.reshape([image_size, image_size, image_size, 1])
                sub_input2_sequence.append(sub_input2)
                count = count + 1
    print(count)


print("count:%d", count)
# Make list to numpy array. With this transform
arrdata1 = np.asarray(sub_input1_sequence, dtype='float32')
arrdata2 = np.asarray(sub_input2_sequence, dtype='float32')
#print(arrdata.shape)
make_data(arrdata1, arrdata2)