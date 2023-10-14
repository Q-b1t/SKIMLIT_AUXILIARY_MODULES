import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from skimlit_dataset_utils.dataset_functions import *
from skimlit_model_versions.model_intance_functions import *
from datetime import datetime

if __name__ == '__main__':
    dataset__root_directory = "pubmed-rct"
    # change this variable accordingly
    dataset_version = "PubMed_20k_RCT_numbers_replaced_with_at_sign"
    dataset_path = os.path.join(dataset__root_directory,dataset_version)

    # model_save_name 
    date = datetime.now().strftime("%m/%d/%Y, %H:%M:%S").replace(",","").replace(" ","_").replace(":","_").replace("/","_")
    model_save_name = f"skimlit_model_{date}."

    # use pretrained embedding layer path
    use_path = "use_pretrained_encoder/universal_sentence_encoder_pretrained.keras"

    # choose the batch size to create the datasets
    BATCH_SIZE = 32

    # fetch dataset training samples
    training_samples = process_text_with_line_numbers(os.path.join(dataset_path,"train.txt"))
    validation_samples = process_text_with_line_numbers(os.path.join(dataset_path,"dev.txt"))
    test_samples = process_text_with_line_numbers(os.path.join(dataset_path,"test.txt"))

    # transform the samples into tables
    raw_data = get_dataset_tables(training_samples,validation_samples,test_samples)

    # get tensorflow prefetched datasets with the corresponding batch size
    prefeched_datasets = get_prefetched_datasets(raw_data,batch_size=BATCH_SIZE)
    training_dataset,validation_dataset,test_dataset = prefeched_datasets["train_char_token_pos_dataset"],prefeched_datasets["validation_char_token_pos_dataset"],prefeched_datasets["test_char_token_pos_dataset"]

    # get training char data
    sentence_data = get_training_sentences(raw_data)
    char_data = get_chars(sentence_data)
    training_chars = char_data["training_chars"]

    # get truncation data
    truncation_data = get_truncation_values(raw_data)

    # instace the model
    skimlit_model = skimlit_model_mk_II(get_hyperparameters(),truncation_data,training_chars,use_path)

    # compile the model
    skimlit_model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = 0.2),
        optimizer = tf.keras.optimizers.Adam(),
        metrics = ["accuracy"]
    )

    skimlit_model.summary()
