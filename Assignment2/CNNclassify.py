import os
import sys
import tensorflow as tf
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

def CNNtrain(x_train, y_train, epochs=10, valid_data=None):
  ''' Using tf.keras module to train a model.
      - The optimizer: Adam
      - Loss function: Sparse categorical crossentropy '''
  
  input_shape = x_train.shape[1:]

  model = tf.keras.Sequential()
  
  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5), strides=(1, 1), activation='relu', input_shape=input_shape))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
  model.add(tf.keras.layers.Dropout(0.2))
  
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
  model.add(tf.keras.layers.Dropout(0.4))
  
  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2, 2), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
  model.add(tf.keras.layers.Dropout(0.2))
  
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(512))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.Dense(512))
  model.add(tf.keras.layers.BatchNormalization())
  model.add(tf.keras.layers.Activation('relu'))
  model.add(tf.keras.layers.Dropout(0.4))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  file_path = "./model/"
  file_dir = os.path.dirname(file_path)
  if not os.path.exists(file_dir):
    os.makedirs(file_dir)
    
  checkpointer = tf.keras.callbacks.ModelCheckpoint('./model/model.ckpt', 
                                                   monitor = 'val_acc',
                                                   verbose=1, 
                                                   save_best_only=True)

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  if valid_data:
    history = model.fit(x_train, 
                        y_train, 
                        epochs=epochs,
                        batch_size=64,
                        validation_data=valid_data,
                        callbacks=[checkpointer])
    return model, history
  else:
    model.fit(x_train, 
              y_train, 
              epochs=epochs,
              batch_size=64,
              validation_data=valid_data)
    model.save('./model/model.ckpt')
    return model
    
def plotHistory(history):
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  
  epochs = range(1, len(acc) + 1)
  
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.plot(epochs, val_acc, 'b', label='Validation acc')
  plt.title('Training and validation accuracy')
  plt.legend()
  
  fig = plt.figure()
  
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()
  plt.show()
  fig.savefig('History.png')
  
def plotFirstConv(new_model, test_img):
  # Names of the layers, so you can have them as part of your plot
  layer_names = []
  for layer in new_model.layers[:12]:
      layer_names.append(layer.name) 
  
  # Extracts the outputs of the top 12 layers
  layer_outputs = [layer.output for layer in new_model.layers[:12]] 
  # Creates a model that will return these outputs, given the model input
  activation_model = tf.keras.models.Model(inputs=new_model.input, outputs=layer_outputs) 
  # Predict one of test set picture
  activations = activation_model.predict(np.reshape(test_img, (1,32,32,3)))
  # Extract first layer output
  first_layer_activation = activations[0]
  
  images_per_col = 6
  images_per_row = 6
  
  # Number of features in the feature map
  n_features = first_layer_activation.shape[-1] 
  #The feature map has shape (1, size, size, n_features)
  feature_size = first_layer_activation.shape[1] 
  display_grid = np.zeros((feature_size * images_per_col, feature_size * images_per_row))
  # Integrate feature maps into one display grid
  for col in range(images_per_col):
    for row in range(images_per_row):
      if (col * images_per_row + row + 1) > n_features:
        break
      channel_image = first_layer_activation[0, :, :, col * images_per_row + row]
      channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
      channel_image /= channel_image.std()
      channel_image *= 64
      channel_image += 128
      channel_image = np.clip(channel_image, 0, 255).astype('uint8')
      # Displays the grid
      display_grid[col * feature_size : (col + 1) * feature_size, 
                   row * feature_size : (row + 1) * feature_size] = channel_image
      
  scale = 1. / feature_size
  fig = plt.figure(figsize=(scale * display_grid.shape[1],
                      scale * display_grid.shape[0]))
  plt.title(layer_names[0])
  plt.grid(False)
  plt.imshow(display_grid, aspect='auto', cmap='gray')
  fig.savefig('CONV_rslt.png')

def load_model():
  ''' Loading an existed model through tf.keras module '''
  
  try:
    print("Loading model from 'model' !!")
    new_model = tf.keras.models.load_model('./model/model.ckpt')
    #new_model.summary()
  except:
    print("Can't load Model !!")
    print("End Program !!")
    sys.exit()
  return new_model

def load_cifar10():
  ''' Loading Cifar10 dataset through tf.keras module '''

  try:
    print("Loading Cifar10 !!")
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  except:
    print("Can't load Cifar10 !!")
    print("End Program !!")
    sys.exit()
  return (x_train, y_train), (x_test, y_test)


label_list = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 
              4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
GPU_RUN = True
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  GPU_RUN = False
  print('GPU device not found. Using CPU !!')
else:
  print('Found GPU at: {}'.format(device_name))


if sys.argv[1] == "train":
  (x_train, y_train), (x_test, y_test) = load_cifar10()
  if GPU_RUN:
    with tf.device('/gpu:0'):
      model, history = CNNtrain(x_train, y_train, epochs=100, valid_data=(x_test, y_test))
  else:
      model = CNNtrain(x_train, y_train, epochs=100, valid_data=(x_test, y_test))
  print("Model saved in file: ./model/model.ckpt")
  
elif sys.argv[1] == "predict" or sys.argv[1] == "test":
  new_model = load_model()
  if GPU_RUN: 
    with tf.device('/gpu:0'):
      test = image.load_img(sys.argv[2], target_size=(32, 32, 3))
      prediction_vec = new_model.predict(np.reshape(test, (1, 32, 32, 3)))
      print(label_list[np.argmax(prediction_vec)])
      plotFirstConv(new_model, test)
  else:
      test = image.load_img(sys.argv[2], target_size=(32, 32, 3))
      prediction_vec = new_model.predict(np.reshape(test, (1, 32, 32, 3)))
      print(label_list[np.argmax(prediction_vec)])
      plotFirstConv(new_model, test)
      
elif sys.argv[1] == "eval":
  (x_train, y_train), (x_test, y_test) = load_cifar10()
  new_model = load_model()
  new_model.evaluate(x_test, y_test)
  
else:
  print("Please try 'train' or 'test' as an argument")