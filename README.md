# pytorch_unet
This project is a pytorch implementation of UNet model.

This project divides training data into train and validation, and test data is used for testing the model. 

The dataset used in this project is from Kaggle. A direct google drive link has been provided to download the dataset. 

The model is trained on the CPU for a limited number of epochs, and the progress of the model is inspected using Weights and Biases. 

The results generated using this model are not accurate since the model is not trained to the convergence.

A Flask app has been created to use the trained UNet model. This app will receive a repository of .tif images, and predictions will be generated and saved in the results repository. 

The app can also be accessed using docker.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contact](#contact)

## Installation

To install and run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/mi161374/pytorch_unet.git`
2. Change into the project directory: `cd pytorch_unet`
3. Install dependencies: `pip install -r requirements.txt`

Please note that Python 3.11.4 was used in this project.

## Usage
1. Download and place the dataset in the main directory: Link: `https://drive.google.com/file/d/16UZRha_PXATKcjTwbYGy6-QW0kPZivPx/view`
2. To train the model and save the weights: `python main.py path_to_training train_val_split learning_rate n_batch n_epochs path_to_save_model`

   example: `python main.py .pytorch_unet\\training 0.8 0.0001 2 2 .pytorch_unet\\unet_model.pth`
4. To generate results and overlay the results on original images, run the development server: `python app.py`
5. Send request: `python request.py`
6. To pull the docker container: `docker pull mi161374/pytorch_unet_flask:latest` 



## Contact

mahdi.imani@unimelb.edu.au
