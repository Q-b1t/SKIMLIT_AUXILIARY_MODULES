import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import string
from tensorflow.keras.models import load_model

alphabet = string.ascii_lowercase + string.digits + string.punctuation
NUM_CHAR_TOKENS = len(alphabet) + 2 # space and OOV token

def get_hyperparameters():
  return {
      "char_embedding_output_dim": 25,
      "token_layer_1_hidden_units":300,
      "char_bidirectional_lstm_hidden_units":50,
      "line_number_dense_1_hidden_units":64,
      "total_line_dense_1_hidden_units":64,
      "char_token_hybrid_dense_1_hidden_units":350,
      "char_token_hybrid_dropout_1_probability":0.5,
      "char_token_hybrid_dense_2_hidden_units":350,
      "tribrid_bidirectional_lstm_hidden_units":200,
      "output_layer_hidden_units":5
  }

def skimlit_model_mk_I(hyperparameters,truncation_params,char_samples):
    """
    This version downloads an USE embedding layer from tensorflow hub on every call.
    This takes a lot of time in the jetson nano. Therefore the training script uses mk_II version.
    This one is ideal for colaboratory or cloud environments with a descent download speed thought.
    """
    char_embedding_output_dim = hyperparameters["char_embedding_output_dim"]
    token_layer_1_hidden_units = hyperparameters["token_layer_1_hidden_units"]
    char_bidirectional_lstm_hidden_units =hyperparameters["char_bidirectional_lstm_hidden_units"]
    line_number_dense_1_hidden_units = hyperparameters["line_number_dense_1_hidden_units"]
    total_line_dense_1_hidden_units = hyperparameters["total_line_dense_1_hidden_units"]
    char_token_hybrid_dense_1_hidden_units = hyperparameters["char_token_hybrid_dense_1_hidden_units"]
    char_token_hybrid_dropout_1_probability = hyperparameters["char_token_hybrid_dropout_1_probability"]
    char_token_hybrid_dense_2_hidden_units = hyperparameters["char_token_hybrid_dense_2_hidden_units"]
    tribrid_bidirectional_lstm_hidden_units = hyperparameters["tribrid_bidirectional_lstm_hidden_units"]
    output_layer_hidden_units = hyperparameters["output_layer_hidden_units"]

    # truncation measures
    line_number_truncation,total_lines_truncation,output_sequence_char_length = truncation_params["line_number_truncation"],truncation_params["total_lines_truncation"],truncation_params["output_sequence_char_length"]

    # instance a pretrained Tensorflow Hub USE
    use_embedding_layer = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",trainable = False,name = "universal_forwarder_encoder")

    # instance embedding layers
    char_vectorizer = layers.TextVectorization(
        max_tokens=NUM_CHAR_TOKENS,
        output_sequence_length = output_sequence_char_length, # pending
        standardize="lower_and_strip_punctuation",
        name = "char_vectorizer"
    )

    char_vectorizer.adapt(char_samples)

    char_vocab = char_vectorizer.get_vocabulary()

    char_embedding = layers.Embedding(
        input_dim = len(char_vocab),
        output_dim = char_embedding_output_dim,
        mask_zero = True,
        name = "char_embedding"
    )
    # build the model

    # token inputs
    token_inputs = layers.Input(shape = [],dtype = "string",name = "token_inputs")
    token_embeddings = use_embedding_layer(token_inputs)
    token_outputs = layers.Dense(token_layer_1_hidden_units,activation = "relu",name = "token_layer_1")(token_embeddings)
    token_model = tf.keras.Model(inputs = token_inputs,outputs = token_outputs)

    # char inputs
    char_inputs = layers.Input(shape = (1,),dtype = "string", name = "char_inputs")
    char_vectors  = char_vectorizer(char_inputs)
    char_embeddings = char_embedding(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(char_bidirectional_lstm_hidden_units),name = "char_bidirectional_lstm")(char_embeddings)
    char_model = tf.keras.Model(inputs = char_inputs,outputs = char_bi_lstm)

    # line numbers model
    line_number_inputs = layers.Input(shape = (line_number_truncation,),dtype = tf.float32,name = "line_number_input")
    x = layers.Dense(line_number_dense_1_hidden_units,activation = "relu",name = "line_number_dense_1")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs = line_number_inputs,outputs = x)

    # total lines model
    total_lines_inputs = layers.Input(shape = (total_lines_truncation,),dtype = tf.float32, name = "total_lines_input")
    y = layers.Dense(total_line_dense_1_hidden_units,activation = "relu",name = "total_line_dense_1")(total_lines_inputs)
    total_lines_model = tf.keras.Model(inputs = total_lines_inputs, outputs = y)

    # combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name = "char_token_hybrid_embedding")([token_model.output,char_model.output])
    z = layers.Dense(char_token_hybrid_dense_1_hidden_units,activation = "relu",name = "char_token_hybrid_dense_1")(combined_embeddings)
    z = layers.Dropout(char_token_hybrid_dropout_1_probability,name = "char_token_hybrid_dropout_1")(z)
    z = layers.Dense(char_token_hybrid_dense_2_hidden_units,activation = "relu",name = "char_token_hybrid_dense_2")(z)

    # combine positional embeddings with combined token and char embeddings
    tribrid_embeddings = layers.Concatenate(name = "char_token_positional_embeddings")([line_number_model.output,total_lines_model.output,z])

    # create output layer
    output_layer = layers.Dense(output_layer_hidden_units,activation = "softmax",name = "output_layer")(tribrid_embeddings)

    # put together model with all the inputs
    model = tf.keras.Model(
        inputs = [
            line_number_model.input,
            total_lines_model.input,
            token_model.input,
            char_model.input
        ],
        outputs = output_layer,
        name = "skimlit_tribrid_embedding_model_preliminary"
        )
    return model

def skimlit_model_mk_II(hyperparameters,truncation_params,char_samples,use_embedding_path):
    """
    This version requires a path to the pretrained use model availabe in use_embedding_instance_function.py. 
    The use instance is provided in a file so the model can easily load it from there rather than downloading it 
    from tensorflow hub's cloud (which takes a lot of time in the jetson nano).
    """
    char_embedding_output_dim = hyperparameters["char_embedding_output_dim"]
    token_layer_1_hidden_units = hyperparameters["token_layer_1_hidden_units"]
    char_bidirectional_lstm_hidden_units =hyperparameters["char_bidirectional_lstm_hidden_units"]
    line_number_dense_1_hidden_units = hyperparameters["line_number_dense_1_hidden_units"]
    total_line_dense_1_hidden_units = hyperparameters["total_line_dense_1_hidden_units"]
    char_token_hybrid_dense_1_hidden_units = hyperparameters["char_token_hybrid_dense_1_hidden_units"]
    char_token_hybrid_dropout_1_probability = hyperparameters["char_token_hybrid_dropout_1_probability"]
    char_token_hybrid_dense_2_hidden_units = hyperparameters["char_token_hybrid_dense_2_hidden_units"]
    tribrid_bidirectional_lstm_hidden_units = hyperparameters["tribrid_bidirectional_lstm_hidden_units"]
    output_layer_hidden_units = hyperparameters["output_layer_hidden_units"]

    # truncation measures
    line_number_truncation,total_lines_truncation,output_sequence_char_length = truncation_params["line_number_truncation"],truncation_params["total_lines_truncation"],truncation_params["output_sequence_char_length"]

    # instance a pretrained Tensorflow Hub USE
    use_embedding_layer = load_model(use_embedding_path,custom_objects={'KerasLayer':hub.KerasLayer}) #hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4",trainable = False,name = "universal_forwarder_encoder")

    # instance embedding layers
    char_vectorizer = layers.TextVectorization(
        max_tokens=NUM_CHAR_TOKENS,
        output_sequence_length = output_sequence_char_length, # pending
        standardize="lower_and_strip_punctuation",
        name = "char_vectorizer"
    )

    char_vectorizer.adapt(char_samples)

    char_vocab = char_vectorizer.get_vocabulary()

    char_embedding = layers.Embedding(
        input_dim = len(char_vocab),
        output_dim = char_embedding_output_dim,
        mask_zero = True,
        name = "char_embedding"
    )
    # build the model

    # token inputs
    token_inputs = layers.Input(shape = [],dtype = "string",name = "token_inputs")
    token_embeddings = use_embedding_layer(token_inputs)
    token_outputs = layers.Dense(token_layer_1_hidden_units,activation = "relu",name = "token_layer_1")(token_embeddings)
    token_model = tf.keras.Model(inputs = token_inputs,outputs = token_outputs)

    # char inputs
    char_inputs = layers.Input(shape = (1,),dtype = "string", name = "char_inputs")
    char_vectors  = char_vectorizer(char_inputs)
    char_embeddings = char_embedding(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(char_bidirectional_lstm_hidden_units),name = "char_bidirectional_lstm")(char_embeddings)
    char_model = tf.keras.Model(inputs = char_inputs,outputs = char_bi_lstm)

    # line numbers model
    line_number_inputs = layers.Input(shape = (line_number_truncation,),dtype = tf.float32,name = "line_number_input")
    x = layers.Dense(line_number_dense_1_hidden_units,activation = "relu",name = "line_number_dense_1")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs = line_number_inputs,outputs = x)

    # total lines model
    total_lines_inputs = layers.Input(shape = (total_lines_truncation,),dtype = tf.float32, name = "total_lines_input")
    y = layers.Dense(total_line_dense_1_hidden_units,activation = "relu",name = "total_line_dense_1")(total_lines_inputs)
    total_lines_model = tf.keras.Model(inputs = total_lines_inputs, outputs = y)

    # combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name = "char_token_hybrid_embedding")([token_model.output,char_model.output])
    z = layers.Dense(char_token_hybrid_dense_1_hidden_units,activation = "relu",name = "char_token_hybrid_dense_1")(combined_embeddings)
    z = layers.Dropout(char_token_hybrid_dropout_1_probability,name = "char_token_hybrid_dropout_1")(z)
    z = layers.Dense(char_token_hybrid_dense_2_hidden_units,activation = "relu",name = "char_token_hybrid_dense_2")(z)

    # combine positional embeddings with combined token and char embeddings
    tribrid_embeddings = layers.Concatenate(name = "char_token_positional_embeddings")([line_number_model.output,total_lines_model.output,z])

    # create output layer
    output_layer = layers.Dense(output_layer_hidden_units,activation = "softmax",name = "output_layer")(tribrid_embeddings)

    # put together model with all the inputs
    model = tf.keras.Model(
        inputs = [
            line_number_model.input,
            total_lines_model.input,
            token_model.input,
            char_model.input
        ],
        outputs = output_layer,
        name = "skimlit_tribrid_embedding_model_preliminary"
        )
    return model