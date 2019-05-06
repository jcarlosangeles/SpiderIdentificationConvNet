import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras.preprocessing import image
from keras.layers import Dropout
from keras.applications import VGG19
import matplotlib.pyplot as plt
import os

def get_session(gpu_fraction=0.8):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
  return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Call function to only use a small part of the GPU and leave space for others to run their projects
KTF.set_session(get_session())


base_dir = './img_scaled'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False
'''
conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
	if layer.name == 'block5_conv1':
		set_trainable = True
	if set_trainable:
		layer.trainable = True
	else:
		layer.trainable = False
'''

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

#class_weights = {0: 3.829338, 1:  1.}

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=32, class_mode='binary')

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

model.save('spiders_12.h5')

test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
