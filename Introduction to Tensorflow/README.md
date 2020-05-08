# Introduction to Tensorflow for AI, ML



### Notes/Highlights

- Know the `optimizers` (Adam, SGD) and `loss functions` (eg: mean square error) of the neural network and their purpose. 

- `tf.keras.Sequential()` stacks layers as a model

  ```python
  # A simple neural network
  # create data
  # TensorFlow v2.2
  xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
  ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
  # create model with layers
  model = tf.keras.Sequential([
      keras.layers.Dense(units=1, input_shape=[1])
  ])
  # compile model with optimizer and loss function
  model.complie(optimizer="sgd", loss_function="mean_square_error")
  # fit the model with epochs
  model.fit(xs, ys, epochs=500)
  # predict
  model.predict([20])
  ```

- You will almost always deal with *probabilities, not certainties*

- Fashion MNIST 28x28 array of greyscales, why the labels are not in English text?

  - Computers understand numbers better
  - Multiple language names can be used to represent a number.
  - It will help us in avoiding BIAS.

- Fashion MNIST,

  - Loss function = `Cross Entropy`

  - 3 layer NN 

    ```python
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                     tf.keras.layers.Dense(128,activation=tf.nn.relu), 
                                  tf.keras.layers.Dense(10,activation=tf.nn.softmax)])
    ```

  - Activation function for last layer: `Softmax` e^yi / sum (e^yi)

  - model.compile() builds/define the model

    ```python
    model.compile(optimizer = tf.optimizers.Adam(),
                  loss = 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    ```

  - Evaluate the model: `model.evaluate(test_images, test_labels)`

  - Predict values: `model.predict(test_images)`

- Callbacks in NN, training a neural network until it reaches a certain threshold. eg: Loss < 0.71.

  ```python
  class myCallback(tf.keras.callbacks.Callback):
  	def on_epoch_end(self, epochs, logs={}):
  		if logs.get("loss") < 0.4:
              print("60% accuracy so stopping the method")
              self.model.stop_training = True
   
  callbacks = myCallback()
  model.fit(optimizer=tf.optimizers.Adam(), # optimizer="Adam"
  		  loss = "sparse_categorical_crossentropy",
            metrics="accuracy",
            callbacks=[callbacks]
           )
  ```

- The initial approach was using Deep neural network with multiple layers like - Flatten, Dense, etc. In the next phase, we will build a classifier for FASHION MNIST using CNN.

- ```python
  training_images=training_images.reshape(60000, 28, 28, 1)
  test_images = test_images.reshape(10000, 28, 28, 1)
  
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  ```

- Convolution layer is used to extract the relevant features before giving it as input to further layers Dense layer. Almost like a image preprocessing step.

- Pooling layer, it preserves the features and reduces the size of image by a factor of pooling layer filer size. eg: 2x2 filter will reduce the original image by a factor of 2.

- The number of convolution filter is determined randomly and it is a good practice to take a multiple of 32. A poorly designed convolution layer may perform even worse than  a DNN.

- The filter size

- Visualizing convolutional layer

- **model.summary()** It shows the model architecture, it even shows how the image dimensions/shape change as it passes through various layers + number of parameters

- **Image generator in tensorflow** :

  ```python
  from tensorflow.keras.preprocessing.image import ImageGenerator
  
  # The directory to the image should be the train/validation folder containing classes 
  # not the sub directory (/project/train and not /project/train/class1)
  
  train_datagen = ImageGenerator(1./255)
  
  # flow from directory to load the images from the dir
  train_generator = train_datagen.flow_from_directory(
      train_dir, # Name of sub directory -> name of class
      target_size = (300,300),
      batch_size = 128, # experiment with various batch sizes to find the imapact of performance
      class_mode = 'binary'
  )
  test_datagen = ImageGenerator(rescale = 1./255) # more at https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/
  test_generator = test_datagen.flow_from_directory(
  	valid_dir,
      target_size= (300,300),
      batch_size= 128, # a batch of images considered while training
      class_mode= "binary"
  )
  ```

- Training

  ```python
  from tensorflow.keras.preprocessing.image import ImageDataGenerator
  
  # All images will be rescaled by 1./255
  train_datagen = ImageDataGenerator(rescale=1/255)
  validation_datagen = ImageDataGenerator(rescale=1/255)
  
  # Number of training images: 1027
  # Flow training images in batches of 128 using train_datagen generator
  train_generator = train_datagen.flow_from_directory(
          '/tmp/horse-or-human/',  # This is the source directory for training images
          target_size=(300, 300),  # All images will be resized to 150x150
          batch_size=128,
          # Since we use binary_crossentropy loss, we need binary labels
          class_mode='binary')
  
  # Number of validation images: 256
  # Flow training images in batches of 128 using train_datagen generator
  validation_generator = validation_datagen.flow_from_directory(
          '/tmp/validation-horse-or-human/',  # This is the source directory for training images
          target_size=(300, 300),  # All images will be resized to 150x150
          batch_size=32,
          # Since we use binary_crossentropy loss, we need binary labels
          class_mode='binary')
  
  history = model.fit(
        train_generator,
        steps_per_epoch=8,  
        epochs=15,
        verbose=1,
        validation_data = validation_generator,
        validation_steps=8)
  ```
