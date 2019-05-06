from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

img_path = 'C:\\Users\\Juan Carlos\\Documents\\10. Décimo Semestre\\Machine Learning\\Python\\DeepLearning\\Spiders\\Results\\Best\\Test\\MalePeacockSpider.jpg'

datagen = ImageDataGenerator(rotation_range=60, width_shift_range=0.2, height_shift_range=0.3, shear_range=0.3, zoom_range=0.3, horizontal_flip=True, fill_mode='nearest')

img = image.load_img(img_path, target_size=(500, 500))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
