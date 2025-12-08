# CAP6415_F25 PROJECT: Kaggle Competition “Google Research - Identify Contrails to Reduce Global Warming”
## Context
This repository contains the files for the final project for the course CAP6415_F25 . The project consists in participating in the Kaggle competition to train a machine learning model to identify contrails in satellite images and help prevent their formation. The repository has two Kaggle notebooks, one for training the model and another for the submission of the competition. For context contrails are condensation trails, which are long, thin ice crystals clouds that form in an aircraft engine when it flies in humid areas.  Some of these contrails can linger and grow for several hours, occasionally merging into cloud formations that become visually identical to naturally occurring cirrus, making them difficult to detect and they can contribute to global warming by trapping heat in the atmosphere. 
### Objective
Train model for image segmentation to generate binary masks to identify which pixels have contrails. The competition is evaluated on the global Dice coefficient. The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. 
## Data
The dataset provided by the challenge consisted of geostationary satellite images retrieved from the GOES-16 Advanced Baseline Imager (ABI). Because contrails are easier to identify with temporal context, a sequence of images at 10-minute intervals are provided. Each example (record_id) contains exactly one labeled frame. The training folder has 20.5k folders representing record id, for each record id, there are 9 folders for each band (band_08 to band_16) and each band folder contains an array that represent sequences of 8 images of size 256x256 representing temporal context, like a video. Each record id also holds a human_pixel_masks.npy, which is a 256x25 array that is the per pixel binary ground truth The ground truth for the contrail detection was determined by 4+ different labelers annotating each image. Pixels were considered a contrail when >50% of the labelers annotated it as such.

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
# Base directory would be "/content/contrails_data" so in the code change DATA_DIR 
# "/kaggle/input/google-research-identify-contrails-reduce-global-warming" for "/content/contrails_data" 
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
**3D U-Net + ConvLSTM** <br>
Proposed a Model consisting of a 3D U-Net with ConvLSTM in Bottle Neck. 3D U-Net is a model that is widely used for 3D image segmentation that consists of 3D convolutional layers and a symmetrical encoder-decoder structure with skip connections to segment volumetric data. In this project the 3D data is provided by the temporal information. Instead of just doing the 3D convolutions in the bottle neck of the U-Net, to better process temporal context, a ConvLSTM model was used to model temporal evolution of contrail formation so the model learns how features evolve over time.<br>
The model was implemented in a Kaggle notebook which is necessary to be able to make a submission for the competition. The programming language used was Python and the model was implemented with Pytorch.<br>
## Training and Submission
The model was trained in an initial Kaggle noteebook for 15 epochs using the GPU T4 x2, using a combine loss function of 0.5 × BCEWithLogitsLoss + 0.5 × Dice Loss. Training time lasted approximately 6 hours. From there the weights were exported as a dataset to be able to use the model in a new notebook for the submission. The submission has scertain requirements, you have to submit a Kaggle notebook, GPU Notebook has to have less than 9 hours run-time, and the internet access disabled. This competition is evaluated on the global Dice coefficient, and the exact format of the submission had to be a run-length encoding on the pixel values.So, you have to submit pairs of values that contain a start position and a run length, e.g., '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).<br>
### Training Kaggle Notebook
https://www.kaggle.com/code/nicolashernandez1307/cap6415-contrail-project-training
### Submission Kaggle Notebook
https://www.kaggle.com/code/nicolashernandez1307/cap6415-contrail-project-submission

## Important references
**Original papers:** <br>
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning dense volumetric segmentation from sparse annotation. arXiv. https://arxiv.org/abs/1606.06650 <br>
- Shi, X., Chen, Z., Wang, H., Yeung, D.-Y., Wong, W.-k., & Woo, W.-c. (2015). Convolutional LSTM network: A machine learning approach for precipitation nowcasting. arXiv. https://arxiv.org/abs/1506.04214 <br>
- LearnOpenCV. (2020). 3D U-Net for BRATS: Implementation & Tutorial. LearnOpenCV. https://learnopencv.com/3d-u-net-brats/ <br>
**Code references:** <br>
- LearnOpenCV. (2020). 3D U-Net for BRATS: Implementation & Tutorial. LearnOpenCV. https://learnopencv.com/3d-u-net-brats/
- Wen, Q. (2020). ConvLSTM PyTorch implementation [Code repository]. GitHub. https://github.com/ndrplz/ConvLSTM_pytorch
<br><br>
For more references used for this project go to: https://github.com/nicolasherna2025/CAP6415_F25_project-Google_Research_Identify_Contrails_Competition/blob/main/References/references.txt


 



