# LFCA-CR-NET
Repository for International Journal of Computer Vision paper "Probabilistic-based Feature Learning of Light Fields for Compressive Imaging and Denoising"

https://trebuchet.public.springernature.app/get_content/6401293d-1745-4b94-9f65-b206fb1b5f3e?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20240112&utm_content=10.1007/s11263-023-01974-9

# Dataset
You can download the dataset for LF denosing from 

https://drive.google.com/drive/folders/1emg1Ll2KPmqkMGuEvLOp7fA6i_kEBYtM?usp=sharing

For the compressive LF imaging, we provide MATLAB code for preparing the training and test data. Please first download light field datasets, and put them into corresponding folders in LFData.


# Requirements
- Python 3.8.8
- PyTorch 1.13.1


# Training

For the tasks of compressive imaging and denoising, it is necessary to pretrain a model initially and then utilize this pretrained model to obtain the final Probabilistic-based Feature Embedding (PFE) model. Let's consider the LFCA as an example.

First, pretrain the model by running 'LFCA-PBF-preTrain.ipynb'.

Next, train the PBF model by running 'LFCA-PBF-Train-Original.ipynb'.

You will need to configure the training data path and set the learning rate according to the type of data you are working with. The batch size can also be adjusted as needed.

# Testing

Set the testing configureation. 

And run LFCA-Test.ipynb or LFDN-Test.ipynb
