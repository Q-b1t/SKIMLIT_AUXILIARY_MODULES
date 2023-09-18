import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

def get_lines(filename):
  with open(filename,"r") as f:
    text = f.readlines()
  f.close()
  return text

def process_text_with_line_numbers(filename):
  input_lines = get_lines(filename)
  abstract_lines = ""
  abstract_samples = list()
  # loop through the lines
  for line in input_lines:
    if line.startswith("###"): # check to see if the line is an ID line
      abstract_id = line
      abstract_lines = ""
    elif line.isspace(): # check to see if line is a new line
      abstract_line_split = abstract_lines.splitlines() # split the abstract into separate strings

      # iterate through eachline in a single anbstract and count them at the same time
      for abstract_line_number,abstract_line in enumerate(abstract_line_split):
        line_data = dict() # create the dictionary
        target_text_split = abstract_line.split("\t")
        line_data["target"] = target_text_split[0]
        line_data["text"] = target_text_split[1].lower()
        line_data["line_number"] = abstract_line_number
        line_data["total_lines"] = len(abstract_line_split)
        abstract_samples.append(line_data)
    else: # the line contains a labelled sentence
      abstract_lines += line

  return abstract_samples

# get a dictionary of dataframes with the training data
def get_dataset_tables(train_samples,validation_samples,test_samples):
  return {
    "raw_training_data": pd.DataFrame(train_samples),
    "raw_validation_data": pd.DataFrame(validation_samples),
    "raw_test_data": pd.DataFrame(test_samples)
  }

def get_dataset_labels(raw_data):
  # retrieve the datasets
  train_df,validation_df,test_df = raw_data["raw_training_data"],raw_data["raw_validation_data"],raw_data["raw_test_data"]

  # instance one hot encoder
  one_hot_encoder = OneHotEncoder(sparse_output = False)

  # return training, validation, and test labels
  return {
    "training_one_hot_labels": one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1)),
    "validation_one_hot_labels": one_hot_encoder.fit_transform(validation_df["target"].to_numpy().reshape(-1, 1)),
    "test_one_hot_labels": one_hot_encoder.fit_transform(test_df["target"].to_numpy().reshape(-1, 1))
  }

# get training sentences
def get_training_sentences(raw_data):
  # retrieve the datasets
  train_df,validation_df,test_df = raw_data["raw_training_data"],raw_data["raw_validation_data"],raw_data["raw_test_data"]
  return {
    "training_sentences":train_df["text"].tolist(),
    "validation_sentences": validation_df["text"].tolist(),
    "test_sentences": test_df["text"].tolist()
  }

# make a function to split sentences into characters
def split_chars(text):
  return " ".join(list(text))

def get_chars(sentence_data):
  # retrive the training sentences
  training_sentences,validation_sentences,test_sentences = sentence_data["training_sentences"],sentence_data["validation_sentences"],sentence_data["test_sentences"]
  # returns the processed chars
  return {
    "training_chars": [split_chars(sentence) for sentence in training_sentences],
    "validation_chars": [split_chars(sentence) for sentence in validation_sentences],
    "test_chars": [split_chars(sentence) for sentence in test_sentences]
  } 

def get_char_length_percentile(training_sentences,percetage = 98):
    char_lens = [len(sentence) for sentence in training_sentences]
    return int(np.percentile(char_lens,95))  

# get the length to truncate line number and total lines embeddings
def get_trunctation_params(train_df,percentage = 98):
  return int(np.percentile(train_df.line_number,percentage)),int(np.percentile(train_df.total_lines,98))

def get_truncation_values(raw_data,percentage = 98):
  sentence_data = get_training_sentences(raw_data)
  line_number_truncation,total_lines_truncation = get_trunctation_params(train_df=raw_data["raw_training_data"],percentage=percentage)
  output_sequence_char_length = get_char_length_percentile(training_sentences=sentence_data["training_sentences"],percetage=percentage)
  return {
    "line_number_truncation":line_number_truncation,
    "total_lines_truncation":total_lines_truncation,
    "output_sequence_char_length":output_sequence_char_length
  }

# get the training, validatoin, and test one hot encoding for the dataset
def get_one_hot_encodings(raw_data):
  # retrieve the datasets
  train_df,validation_df,test_df = raw_data["raw_training_data"],raw_data["raw_validation_data"],raw_data["raw_test_data"]

  # get trunctation values
  line_number_truncation, total_lines_trunctation = get_trunctation_params(train_df=train_df)
  
  # get one hot encodings for the line numbers
  train_line_numbers_one_hot = tf.one_hot(train_df["line_number"].to_numpy(),depth = line_number_truncation)
  validation_line_numbers_one_hot = tf.one_hot(validation_df["line_number"].to_numpy(),depth = line_number_truncation)
  test_line_numbers_one_hot = tf.one_hot(test_df["line_number"].to_numpy(),depth = line_number_truncation)

  # get one hot encodings for the total lines
  train_total_lines_one_hot = tf.one_hot(train_df["total_lines"].to_numpy(),depth = total_lines_trunctation)
  validation_total_lines_one_hot = tf.one_hot(validation_df["total_lines"].to_numpy(),depth = total_lines_trunctation)
  test_total_lines_one_hot = tf.one_hot(test_df["total_lines"].to_numpy(),depth = total_lines_trunctation)
  return {
    "line_number": {
      "train":train_line_numbers_one_hot,
      "validation":validation_line_numbers_one_hot,
      "test":test_line_numbers_one_hot
    },
    "total_lines": {
      "train":train_total_lines_one_hot,
      "validation":validation_total_lines_one_hot,
      "test":test_total_lines_one_hot
    }
  }

def get_prefetched_datasets(raw_data,batch_size = 32):
  # retrieve the labels
  dataset_one_hot_labels = get_dataset_labels(raw_data=raw_data)
  training_one_hot_labels, validation_one_hot_labels, test_one_hot_labels = dataset_one_hot_labels["training_one_hot_labels"], dataset_one_hot_labels["validation_one_hot_labels"], dataset_one_hot_labels["test_one_hot_labels"]

  # retrieve the training sentences
  sentence_data = get_training_sentences(raw_data = raw_data)
  training_sentences,validation_sentences,test_sentences = sentence_data["training_sentences"],sentence_data["validation_sentences"],sentence_data["test_sentences"]

  # retriebe the chars
  char_data = get_chars(sentence_data=sentence_data)
  training_chars,validation_chars,test_chars = char_data["training_chars"],char_data["validation_chars"],char_data["test_chars"]

  # get one hot encodingd
  one_hot_encodings = get_one_hot_encodings(raw_data= raw_data)

  # line numbers one hot label
  training_line_numbers_one_hot = one_hot_encodings["line_number"]["train"]
  validation_line_numbers_one_hot = one_hot_encodings["line_number"]["validation"]
  test_line_numbers_one_hot = one_hot_encodings["line_number"]["test"]

  # total lines one hot label
  training_total_lines_one_hot = one_hot_encodings["total_lines"]["train"]
  validation_total_lines_one_hot = one_hot_encodings["total_lines"]["validation"]
  test_total_lines_one_hot = one_hot_encodings["total_lines"]["test"]

  # create prefetch training dataset
  train_char_token_pos_data = tf.data.Dataset.from_tensor_slices((training_line_numbers_one_hot,training_total_lines_one_hot,training_sentences,training_chars))
  train_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(training_one_hot_labels)
  train_char_token_pos_dataset = tf.data.Dataset.zip((train_char_token_pos_data,train_char_token_pos_labels))
  train_char_token_pos_dataset = train_char_token_pos_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  # create prefetch validation dataset
  validation_char_token_pos_data = tf.data.Dataset.from_tensor_slices((validation_line_numbers_one_hot,validation_total_lines_one_hot,validation_sentences,validation_chars))
  validation_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(validation_one_hot_labels)
  validation_char_token_pos_dataset = tf.data.Dataset.zip((validation_char_token_pos_data,validation_char_token_pos_labels))
  validation_char_token_pos_dataset = validation_char_token_pos_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  # create prefetch test dataset
  test_char_token_pos_data = tf.data.Dataset.from_tensor_slices((test_line_numbers_one_hot,test_total_lines_one_hot,test_sentences,test_chars))
  test_char_token_pos_labels = tf.data.Dataset.from_tensor_slices(test_one_hot_labels)
  test_token_pos_dataset = tf.data.Dataset.zip((test_char_token_pos_data,test_char_token_pos_labels))
  test_char_token_pos_dataset = test_token_pos_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

  return {
    "train_char_token_pos_dataset": train_char_token_pos_dataset,
    "validation_char_token_pos_dataset": validation_char_token_pos_dataset,
    "test_char_token_pos_dataset": test_char_token_pos_dataset
  }