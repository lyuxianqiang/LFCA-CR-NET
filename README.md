# LFCA-CR-NET
Repository for International Journal of Computer Vision paper "Probabilistic-based Feature Learning of Light Fields for Compressive Imaging and Denoising"
https://trebuchet.public.springernature.app/get_content/6401293d-1745-4b94-9f65-b206fb1b5f3e?utm_source=rct_congratemailt&utm_medium=email&utm_campaign=nonoa_20240112&utm_content=10.1007/s11263-023-01974-9

# Dataset
You can download the dataset for synthetic LF data from 

# Requirements
- Python 3.8.8
- PyTorch 1.13.1


# Training
Set the training datapath, and learning rate according to data type. You can also change the batchsize accordingly. 

And run 'python train_synf.py'

When training on the L3F dataset, it is advisable to configure the learning rate to 5e-4 for L3F-20, 1e-4 for L3F-50, and 1e-4 for L3F-100, respectively.

For other datasets, we suggest adapting the learning rate selection strategy based on the average brightness level of the dataset. Specifically, a lower learning rate is recommended for datasets with darker overall brightness.

# Testing
Set the testing datapath. 

And run 'python test_synf.py'
