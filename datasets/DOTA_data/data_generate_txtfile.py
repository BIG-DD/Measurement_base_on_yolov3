import os
import random

img_path = './images'
imgtxt_path = './ImageSets'
total_xml = os.listdir(img_path)

txt_train = open(os.path.join(imgtxt_path,'train.txt'), 'w')
txt_val = open(os.path.join(imgtxt_path,'val.txt'), 'w')
txt_test = open(os.path.join(imgtxt_path,'test.txt'), 'w')
len_img = []

for i in range(len(total_xml)):
    len_img.append(i)
random.shuffle(len_img)

for ii in range(len(total_xml)):
    if ii < (len(total_xml)*0.8):
        name = './datasets/DOTA_data/images/' + total_xml[len_img[ii]] + '\n'
        txt_train.write(name)
    if (len(total_xml) * 0.8) <= ii < (len(total_xml) * 0.9):
        name = './datasets/DOTA_data/images/' + total_xml[len_img[ii]] + '\n'
        txt_val.write(name)
    if (len(total_xml) * 0.9) <= ii <= (len(total_xml) * 1):
        name = './datasets/DOTA_data/images/' + total_xml[len_img[ii]] + '\n'
        txt_test.write(name)