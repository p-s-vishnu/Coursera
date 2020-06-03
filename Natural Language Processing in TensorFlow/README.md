# NPL in Tensor flow

[TOC]



#### Week 1 - Sentiment in Text

- Word based encodings, why ASCII cannot be used to represent embedding? 
  Even though it is a good approach but it doesn't preserve the semantics of the word *eg:  semantics: SILENT AND LISTEN*

- Tokenizing words

  ```python
  # from namespace import module
  from tensorflow.keras.preprocessing.text import Tokenizer
  from tensorflow.keras.preprocessing.sequence import Padding
  
  sentences = ["asd", "this is for text","Another text!"]
  
  # num_words specifies top words it will encode
  # There is a trade off between num_words and the training time
  tokenizer = Tokenizer(num_words=100, 
                        oov_token="<OOV>") # oov_token will replace with unseen sentences
  tokenizer.fit_on_texts(sentences)
  
  # will return the key value pair of word and its corresponding embedding, Punctuations will be removed and the keys will be converted to lower
  word_indexes = tokenizer.word_index 

  # num_words specified is reflected in the text_to_sequence process and not word_indexes
  sequence = tokenizer.texts_to_sequences(sentences)

  # By default appropriate number of 0 before the index
  padded = pad_sequence(sequence, padding="post",
                        trucating="post", max_len=5)

  ```
  
- Resource: Removing stopwords using [NLTK, spacy or genism](https://stackabuse.com/removing-stop-words-from-strings-in-python/) .

#### Week 2 - Word Embeddings

Embeddings, where the tokens are mapped as **vectors** in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space.

- pip install tensorflow-datasets, sample datasets in the package
  <img src="C:\Repos\github\Coursera\Natural Language Processing in TensorFlow\Week 2\Datasets in tensorflow.png" alt="tensorflow-dataset" style="zoom:80%;" />

- Check the tensorflow version, if it is 1.X version you will have to enable the eager execution by default.

  ```python
  import tensorflow_datasets as tfds
  
  imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
  train_data, test_data = imdb['train'], imdb['test'] # 25,000 each
  
  train_sentences = []
  train_labels = []
  test_sentences = []
  test_labels = []
  # <type numpy>.numpy() will extract their values
  # in python 3 str(s.numpy()) is required instead of just .numpy()
  for s,l in train_data:
      train_sentences.append(str(s.numpy()))
      train_labels.append(l.numpy())
  # simmilarly for test set	
      
  #hyperparameters
  vocab_size=10000
  embedding_dim = 16
  max_length= 120
  trunc_type = "post"
  padding_type = "post"
  oov_token = "<OOV>"
  
  # main work
  tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
  # extract word index
  tokenizer.fit_on_texts(train_sentences)
  sequences = tokenizer.texts_to_sequences(training_sentences)
  padding = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
  
  testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
  testing_padding = pad_sequences(testing_sequence, maxlen=max_length)
  
  # Neural network
  model = tf.keras.sequential([
          tf.keras.layers.Embedding(vocab_size, embedding_dim, input_layer=max_length),
          # tf.keras.layers.Flatten(),
      	tf.keras.layers.GlobalAveragePooling1D(),
          tf.keras.layers.Dense(6, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
      ])
  
  model.compile(loss="binary_cross_entropy", optimizer="adam", metrics=["accuracy"])
  history = model.fit(train_data, epochs=num_epochs, validation_data=test_data)
  ```

- Check the train and test for accuracy and loss graphs while tuning the hyperparameters to check for overfitting or underfitting.

- Pre tokenized dataset, how sequence of words can be as important as the words
  https://github.com/tensorflow/datasets/tree/master/docs/catalog

  ```
  imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
  tokenizer = info.features['text'].encoder
  ```

- [Subwords encoder](https://www.tensorflow.org/datasets/api_docs/python/tfds/features/text/SubwordTextEncoder)

- [Tensorflow dataset](https://www.tensorflow.org/datasets/catalog/overview)

-  **Looked at taking your tokenized words and using Embeddings to establish meaning from them in a mathematical way**

- **Words were mapped to vectors in higher dimensional space, and the semantics of the words then learned when those words were labelled with similar meaning. So, for example, when looking at movie reviews, those movies with positive sentiment had the dimensionality of their words ending up ‘pointing’ a particular way, and those with negative sentiment pointing in a different direction.** 

- **From this, the words in future sentences could have their ‘direction’ established, and from this the sentiment inferred. You then looked at sub word tokenization, and saw that not only do the meanings of the words matter, but also the sequence in which they are found.**

#### Week 3 - Sequence models

- RNN helps in maintaining the sequence whereas a LSTM even preserves the context of sequence

  ```python
  model = tf.keras.Sequential([
          tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim=64),
      	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64) ), 
      # Add return_sequence=True if you want stacked lstm => the output of the previous layer matches with the input of the next layer
      # 	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
          tf.keras.layers.Dense(6, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
      ])
  ```

- Try comparing the results of an LSTM with Basic Flatten and Global Average Pooling layer.

- Convolutional network in text

  ```python
  model = tf.keras.Sequential([
          tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim=64, input_length=max_length),
      	tf.keras.layers.Convo1D(128, 5, activation="relu"),
  		tf.keras.layers.GlobalAveragePooling1D(),
        	tf.keras.layers.Dense(26, activation='relu'),
          tf.keras.layers.Dense(1, activation='sigmoid'),
  ```

  

- Exploring overfitting in NLP

  One of the major reasons for the overfitting being that your training dataset was quite small, and with a small number of words. Embeddings derived from this may be over generalized also. So for this week’s exercise you’re going to train on a large dataset, as well as using transfer learning of an existing set of embeddings.
  Dataset: https://www.kaggle.com/kazanova/sentiment140

  The embeddings that you will transfer learn from are called the GloVe, also known as Global Vectors for Word Representation, available at: https://nlp.stanford.edu/projects/glove/

#### Week 4 - Sequence models and literature

- Create/Predict next word


  ```python
input_sequence = []

for line in corpus:
	token_list = token.texts_to_sequence([line])[0]
    for i, _ in enumerate(token_list):
        input_sequence.append(token_list[:i+1])
  ```

- As the number of predicted word increases the output can become gibberish. Reason, the probability that each word matches an existing phrase decreases as the more words you create.
  Optimization tips:
  - You can increase the corpus size
  - Dimensionality of Embeddings
  - Number of LSTM cells
  - Toggling between bidirectional and unidirectional LSTM
  -  Optimizer, eg: Adam( lr = 0.01)
- Final project has , 1696 sentences in all.
- The drawback of using word based encoding is that there are far more words in a corpus than the set of characters which makes it quite memory intensive (Especially during the categorical encoding phase).
- Character based prediction using RNNs https://www.tensorflow.org/tutorials/text/text_generation