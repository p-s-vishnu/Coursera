# NPL in Tensor flow

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
  
  # will return the key value pair of word and its corresponding embedding 
# Punctuations will be removed and the keys will be converted to lower
  word_indexes = tokenizer.word_index 
  
  # num_words specified is reflected in the text_to_sequence process and not word_indexes
  sequence = tokenizer.texts_to_sequences(sentences)
  
  # By default appropriate number of 0 before the index
  padded = pad_sequence(sequence, padding="post",
                        trucating="post", max_len=5)
  
  ```
  
- Resource: Removing stopwords using [NLTK, spacy or genism](https://stackabuse.com/removing-stop-words-from-strings-in-python/) .

#### Week 2 - Word Embeddings

Embeddings, where the tokens are mapped as vectors in a high dimension space. With Embeddings and labelled examples, these vectors can then be tuned so that words with similar meaning will have a similar direction in the vector space.

- 

#### Week 3 - Sequence models



#### Week 4 - Sequence models and literature

