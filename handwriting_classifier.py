import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import matplotlib.pyplot as plt
import numpy as np
import random

def get_label_color(val1, val2):
  if val1 == val2:
    return 'black'
  else:
    return 'red'

pixel_width = 28
pixel_height = 28
no_of_classes = 10
batch_size = 32
epochs = 10

(features_train, labels_train), (features_test, labels_test) = mnist.load_data()
features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height, 1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height, 1)

input_shape = (pixel_width, pixel_height, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

features_train /= 255
features_test /= 255

labels_train = keras.utils.to_categorical(labels_train, no_of_classes)
labels_test = keras.utils.to_categorical(labels_test, no_of_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(no_of_classes, activation='softmax'))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(features_train, labels_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(features_test, labels_test))

score = model.evaluate(features_test, labels_test, verbose=0)

predictions = model.predict(features_test)
prediction_digits = np.argmax(predictions, axis=1)

plt.figure(figsize=(18, 18))
for i in range(100):
  ax = plt.subplot(10, 10, i+1)
  plt.xticks([])
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  image_index = random.randint(0, len(prediction_digits))
  plt.imshow(np.squeeze(features_test[image_index]), cmap=plt.cm.gray)

  ax.xaxis.label.set_color(get_label_color(prediction_digits[image_index],
                                           np.argmax(labels_test[image_index])))
  #print(image_index)
  plt.xlabel('Predicted: %d' % prediction_digits[image_index])
plt.show()
