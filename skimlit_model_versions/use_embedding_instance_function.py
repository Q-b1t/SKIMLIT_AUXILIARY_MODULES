import tensorflow_hub as hub
import tensorflow as tf
from tensorflow.keras import layers, Model

"""
Download and build a universal sentence encoder model instance. This is only for backup
purposes. The model will use a downloaded use instance stored in this directory since
it takes a lot of time for it to be downloaded on every run of the training script. 
"""
def get_universal_sentence_encoder():
  use_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",trainable = False,name = "universal_forwarder_encoder")
  token_inputs = layers.Input(shape = [],dtype = "string",name = "token_inputs")
  token_embeddings = use_embedding_layer(token_inputs)
  use_model = Model(inputs = token_inputs,outputs = token_embeddings,name = "use_embedding_layer")
  return use_model
