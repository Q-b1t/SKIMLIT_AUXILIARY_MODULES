import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from skimlit_dataset_utils.dataset_functions import *
from skimlit_model_versions.model_intance_functions import *

if __name__ == '__main__':
    dataset__root_directory = "../pubmed-rtc"
    # change this variable accordingly
    dataset_version = "PubMed_20k_RCT_numbers_replaced_with_at_sign"
    dataset_path = os.path.join(dataset__root_directory,dataset_version)
    print(dataset_path)
