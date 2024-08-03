# importing system libraries
import re
import pickle
import gensim
import operator
import unicodedata
from string import punctuation
from random import shuffle
from nltk.corpus import stopwords #import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Layer
from tensorflow.keras import initializers

 nltk.download('wordnet')
 nltk.download('stopwords')
 nltk.download('punkt')

class CustomIndependentDense(Layer):
    def __init__(self, activation=None, **kwargs):
        super(CustomIndependentDense, self).__init__(**kwargs)
        self.activation = activation

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1],),
            initializer=initializers.GlorotUniform(),
            trainable=True
        )

    def call(self, inputs):
        z = inputs * self.kernel
        if self.activation == 'heaviside':
            return straight_through_step(z)  # Use custom gradient function
        return z

# Define the custom gradient function for STE
@tf.custom_gradient
def straight_through_step(x):
    def grad(dy):
        # Gradient is 1.0 for the straight-through estimator
        return dy
    return tf.where(x > 0, 1.0, 0.0), grad

class ElementWiseMultiply(Layer):
    def call(self, inputs):
        input_data, custom_layer_output = inputs
        return input_data * custom_layer_output


class feature_extraction:
  def __init__(self):
    with open('/content/tfidf.pkl', 'rb') as f:
      self.tfidf = pickle.load(f)

    # Register the custom layers
    get_custom_objects().update({
      'CustomIndependentDense': CustomIndependentDense,
      'ElementWiseMultiply': ElementWiseMultiply
      })

    # Load the model with custom objects
    self.model = load_model('/content/nn_fs.h5', custom_objects={
      'CustomIndependentDense': CustomIndependentDense,
      'ElementWiseMultiply': ElementWiseMultiply
    })

    self.model = load_model('/content/nn_fs.h5')

    #stopwords
  def stopword(self, str):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(str)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

  def casefolding(self, s):
    new_str = s.lower()  
    return new_str

  def cleaning(self, str):
    #remove digit from string
    str = re.sub("\S*\d\S*", "", str).strip()
    #removeHashtag
    str = re.sub('#[^\s]+','',str)
    #remove mention
    str = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",str)
    #remove non-ascii
    str = unicodedata.normalize('NFKD', str).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    #to lowercase
    str = str.lower()
    #Remove additional white spaces
    str = re.sub('[\s]+', ' ', str)
    return str
  def preprocess(self, data):
    # tokenization
    tokens = self.casefolding(data)
    tokens = self.cleaning(tokens)
    tokens = "".join(tokens)
    
    # listing stopwords from NLTK
    stops = self.stopword(tokens)
    # lemmatization
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w, pos='a') for w in stops]
    
    tokens = "".join(tokens)
    return tokens

  def feature_selection(self, text):

    text = self.preprocess(text)
    # Transform the new text into TF-IDF representation
    text_tfidf = self.tfidf.transform([text])

    # Make a prediction
    prediction = self.model.predict(text_tfidf.toarray())

    probability_score = prediction

    # Threshold the prediction to get the class
    predicted_class = 1 if prediction > 0.5 else 0

    # Create an intermediate model to get the output of CustomIndependentDenseSTE layer
    input_layer = self.model.input
    custom_layer = self.model.get_layer('custom_independent_dense_1').output
    intermediate_model = tf.keras.Model(inputs=input_layer, outputs=custom_layer)

    # Predict the output using the full model
    full_model_output = self.model.predict(text_tfidf.toarray())

    # Get the output of the CustomIndependentDenseSTE layer
    custom_layer_output = intermediate_model.predict(text_tfidf.toarray())
    #print("CustomIndependentDenseSTE Layer Output:", custom_layer_output)
    custom_layer_output = custom_layer_output.flatten().tolist()

    # Get feature as a list
    feature_names = self.tfidf.get_feature_names_out()

    # get vocabulary
    # Tokenize the text by splitting on whitespace
    words = text.split()
    
    # Convert to lowercase to ensure case-insensitivity
    words = [word.lower() for word in words]
    
    # Use a set to get unique words
    vocabulary = set(words)

    selected_features = []
    non_selected_features = []

    for fe, col in zip(custom_layer_output, feature_names):
      if fe != 0:
        selected_features.append(col)
      elif col in vocabulary:
        non_selected_features.append(col)
    return predicted_class, probability_score[0, 0], selected_features, non_selected_features
