import numpy as np
import random
import shutil
from shutil import copyfile
import os

imgs_dangerous = "C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\img_full\\test\\dangerous\\"
imgs_harmless = "C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\img_full\\test\\harmless\\"

tests_path = "C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\statistical_sign\\"

idx = 0

for dir in list(os.walk(tests_path)):
    print(dir)
    directory = dir[0]
    #Copy 30 files form img
    files_dangerous = os.listdir(imgs_dangerous)[idx * 30:30 * idx + 1]
    files_harmless = os.listdir(imgs_harmless)[idx * 30:30 * idx + 1]
    #go back to the dir
    dst_dangerous = os.path.join(directory[idx], 'dangerous')
    print(dst_dangerous)
    copyfile(files_dangerous, dst_dangerous)

    dst_harmless = os.path.join(directory[idx], 'harmless')
    copyfile(files_harmless, dst_harmless)

    idx += 1
