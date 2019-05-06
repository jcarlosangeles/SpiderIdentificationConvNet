from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import numpy as np
import sklearn
from PIL import Image
import matplotlib.pyplot as plt

model = VGG19()
imagePath = 'C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\img_full\\test\\harmless\\29445.harmless.jpg'
test_image = image.load_img(imagePath, target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

image = preprocess_input(test_image)
yhat = model.predict(image)
label = decode_predictions(yhat)
label = label[0][0]
print('{} ({})'.format(label[1], label[2] * 100))

import keract

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
activations = keract.get_activations(model, image)
first = activations.get('block1_conv1/Relu:0')
keract.display_activations(activations)
