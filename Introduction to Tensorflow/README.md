# Introduction to Tensorflow for AI, ML



### Notes/Highlights

- Know the `optimizers` (Adam, SGD) and `loss functions` (eg: mean square error) of the neural network. 

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

