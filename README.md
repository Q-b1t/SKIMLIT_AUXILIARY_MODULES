# Skimlit Auxiliary
Auxiliary functions for the skimlit model escalation and training. 
# Dataset
The dataset used on is the [Pubmed RTC Dataset](https://arxiv.org/abs/1710.06071). In order to download it, one can run the ```download_pumbed_rtc.sh``` script.
# Training Environment 
The training script and the python dependencies are made to train the model on a Jetson Nano. For installing Tensorflow2 in the Jetson Nano refet to [NVIDIA's Documentation](https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html) on the topic. 
The most important requirement is that **the Tensorflow package available to NVIDIA's GPUs is built for python3.8**. 
## A note regarding USE
Furthermore, the Universal Sentence Encoder pretrained embedding is downloaded from Tensorflow Hub. This resulted quite problematic since each time the training script was run it downloaded the layer from the cloud consuming an enormous amount of time. The workaround I came up with is to implement two functions:
- ```skimlit_model_mk_I```: Downloads the USE layer from Tensorflow Hub each time it is called in order to create the model. Useful for cloud environments like Google Colab but suboptimal for the Jetson Nano.
- ```skimlit_model_mk_II```: It takes as argument the path to the USE model file (it can be generated using the function available in ```use_embedding_instance_function.py```) and incorporates it into the architecture. The idea is to train the model on the GPU using this function, save the weights, and load them into an instance of ```skimlit_model_mk_I``` at deployment. 

Since the tensor shapes, names, and layers match, the models are equivalent regardless of the function used to generate them. Therefore the deployment model should accept the weights of the training model.

# References
- [Skimlit](https://arxiv.org/abs/1612.05251)
- [Skimlit Architecture](https://arxiv.org/abs/1710.06071)