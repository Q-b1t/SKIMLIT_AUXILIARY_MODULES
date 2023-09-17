import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import string

alphabet = string.ascii_lowercase + string.digits + string.punctuation
NUM_CHAR_TOKENS = len(alphabet) + 2 # space and OOV token

def skimlit_model_mk_I(truncation_params):

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

    char_vocab = char_vectorizer.get_vocabulary()

    char_embedding = layers.Embedding(
        input_dim = len(char_vocab),
        output_dim = 25,
        mask_zero = True,
        name = "char_embedding"
    )
    # build the model

    # token inputs
    token_inputs = layers.Input(shape = [],dtype = "string",name = "token_inputs")
    token_embeddings = use_embedding_layer(token_inputs)
    token_outputs = layers.Dense(128,activation = "relu")(token_embeddings)
    token_model = tf.keras.Model(inputs = token_inputs,outputs = token_outputs)

    # char inputs
    char_inputs = layers.Input(shape = (1,),dtype = "string", name = "char_inputs")
    char_vectors  = char_vectorizer(char_inputs)
    char_embeddings = char_embedding(char_vectors)
    char_bi_lstm = layers.Bidirectional(layers.LSTM(24))(char_embeddings)
    char_model = tf.keras.Model(inputs = char_inputs,outputs = char_bi_lstm)

    # line numbers model
    line_number_inputs = layers.Input(shape = (line_number_truncation,),dtype = tf.float32,name = "line_number_input")
    x = layers.Dense(34,activation = "relu")(line_number_inputs)
    line_number_model = tf.keras.Model(inputs = line_number_inputs,outputs = x)

    # total lines model
    total_lines_inputs = layers.Input(shape = (total_lines_truncation,),dtype = tf.float32, name = "total_lines_input")
    y = layers.Dense(32,activation = "relu")(total_lines_inputs)
    total_lines_model = tf.keras.Model(inputs = total_lines_inputs, outputs = y)

    # combine token and char embeddings into a hybrid embedding
    combined_embeddings = layers.Concatenate(name = "char_token_hybrid_embedding")([token_model.output,char_model.output])
    z = layers.Dense(256,activation = "relu")(combined_embeddings)
    z = layers.Dropout(0.5)(z)

    # combine positional embeddings with combined token and char embeddings
    tribrid_embeddings = layers.Concatenate(name = "char_token_positional_embeddings")([line_number_model.output,total_lines_model.output,z])

    # create output layer
    output_layer = layers.Dense(5,activation = "softmax",name = "output_layer")(tribrid_embeddings)

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