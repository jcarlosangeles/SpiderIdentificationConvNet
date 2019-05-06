from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from numpy import savetxt
import numpy as np
import os

model = models.load_model('spiders_1_franken.h5')  #model to be loaded
model.compile(optimizer=optimizers.RMSprop(lr=3e-6), loss = 'binary_crossentropy', metrics = ['accuracy'])
test_datagen = ImageDataGenerator(rescale=1./255)

tests_path = 'C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\statistical_sign\\'
results = list()

for elt in os.listdir(tests_path):
    test_dir = os.path.join(tests_path, elt)
    if os.path.isdir(test_dir):
        test_generator = test_datagen.flow_from_directory(test_dir, target_size=(200, 200), batch_size=30, class_mode='binary')
        test_loss, test_acc = model.evaluate_generator(test_generator, steps=1)
        results.append(test_acc)

savetxt('results2.csv', results)
print(results)
