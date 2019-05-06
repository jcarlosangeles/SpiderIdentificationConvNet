import urllib.request
import pandas as pd
import numpy as np
import os
from time import sleep

def downloadImg(url, species):
    dir = os.path.join(dataset_path, species)
    os.chdir(dir)
    idx = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
    img_name = str(idx) + '.' + species + '.jpg'
    urllib.request.urlretrieve(url, img_name)

data = pd.read_csv("Centruroides.csv", sep=';')
print(data)
dataset_path = "C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\Medical Concern\\Centruroides"

if (not os.path.isdir(dataset_path)):
    print('Creating Base Directory...')
    os.mkdir(dataset_path)

species = np.unique(data.values[:,2])

for sp in species:
    species_dir = os.path.join(dataset_path, sp)
    if (not os.path.isdir(species_dir)):
        print('Creating Species Directory...')
        os.mkdir(species_dir)

for sp in range(len(data)):
    img_url = data.values[sp, 0]
    species = data.values[sp, 2]

    try:
        downloadImg(img_url, species)
    except:
        print('Error on img: ' + img_url)

    if (sp % 10 == 0):

        print(str(sp) + ' images downloaded...')
while 1:
    print('Done Downloading!!!')
    print('\a')
    sleep(2)
