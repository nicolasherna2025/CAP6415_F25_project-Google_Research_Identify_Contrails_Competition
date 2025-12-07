# CAP6415_F25 PROJECT: Kaggle Competition “Google Research - Identify Contrails to Reduce Global Warming”
## Context
This repository contains the files for the final project for the course CAP6415_F25 . The project consists in participating in the Kaggle competition to train a machine learning model to identify contrails in satellite images and help prevent their formation. The repository has two Kaggle notebooks, one for training the model and another for the submission of the competition. For context contrails are condensation trails, which are long, thin ice crystals clouds that form in an aircraft engine when it flies in humid areas.  Some of these contrails can linger and grow for several hours, occasionally merging into cloud formations that become visually identical to naturally occurring cirrus, making them difficult to detect and they can contribute to global warming by trapping heat in the atmosphere. 
### Objective
Train model for image segmentation to generate binary masks to identify which pixels have contrails. The competition is evaluated on the global Dice coefficient. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. 
## Data
The dataset provided by the challenge consisted of geostationary satellite images retrieved from the GOES-16 Advanced Baseline Imager (ABI). Because contrails are easier to identify with temporal context, a sequence of images at 10-minute intervals are provided. Each example (record_id) contains exactly one labeled frame.

### Accessing Data
To access the data first you have to accept the competition rules in https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/rules.
#### Kaggle notebook
From a Kaggle notebook you can access the data by going to the Input section - "Add Input" and search for Google Research - Identify Contrails to Reduce Global Warming.
#### Colab
For colab you need to use Kaggle API token with the following code
```
import os
# 1. Update the kaggle library
!pip install -q --upgrade kaggle
# 2. Set the credentials
os.environ['KAGGLE_API_TOKEN'] = "replace with token"
# 3. Download the dataset
!kaggle competitions download -c google-research-identify-contrails-reduce-global-warming
# 4. Unzip the data
!unzip -q google-research-identify-contrails-reduce-global-warming.zip -d /content/contrails_data
!rm google-research-identify-contrails-reduce-global-warming.zip
# Base directory would be "/content/contrails_data"
```
**For more information about context and the data set go to the competition information page in Kaggle: https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming/overview**

## Requirements
torch>=2.0.0 <br>
numpy <br>
pandas <br>
matplotlib <br>
tqdm <br>
GPU <br>

## Solution
**3D U-Net with ConvLSTM in Bottle Neck**
Proposed a Model consisting of a 3D U-Net with ConvLSTM in Bottle Neck. 3D U-Net is a model that is widely used for 3D image segmentation that consists of 3D convolutional layers and a symmetrical encoder-decoder structure with skip connections to segment volumetric data. In this project the 3D data is provided by the temporal information. Instead of just doing the 3D convolutions in the bottle neck of the U-Net, to better process temporal context, a ConvLSTM model was used to model temporal evolution of contrail formation so the model learns how features evolve over time.


