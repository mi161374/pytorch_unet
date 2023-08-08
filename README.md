# pytorch_unet
This project is a pytorch implementation of UNet model.
This project divides training data into train and validation, and test data is used for testing the model. 
The model is trained on the CPU for a limited number of epochs, and the progress of the model is inspected using Weights and Biases. 
The results generated using this model are not accurate since the model is not trained to the convergence.

A Flask app has been created to use the trained UNet model in a none isolated model server. The app will run on port 5000. This app will receive a repository of .tif images, and predictions will be generated and saved in the results repository. 

A docker container can be built using the Dockerfile provided in this project.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Example of Weights and Biases](#Example_of_WeightsandBiases)
- [Contact](#contact)

## Installation

To install and run the project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/username/my-awesome-project.git`
2. Change into the project directory: `cd pytorch_unet`
3. Install dependencies: `pip install -r requirements.txt`

Please note that Python 3.11.4 was used in this project.

## Usage

1. To train the model and save the weights: `python main.py path_to_training train_val_split learning_rate n_batch n_epochs path_to_save_model`
   example: `python main.py .pytorch_unet\\training 0.8 0.0001 2 2 .pytorch_unet\\unet_model.pth`
2. To generate results and overlay the results on original images, run the development server: `python app.py`
3. Send request: `python request.py`
4. To build the docker container: `docker build -t pytorch_unet .`
5. To run the docker container: `docker run -p 5000:5000 pytorch_unet` 

## Example_of_WeightsandBiases:

![Screenshot 2023-08-08 121133](https://github.com/mi161374/pytorch_unet/assets/70301469/32667ff5-26d1-4a68-85e4-1779cf554bb6)

![Screenshot 2023-08-08 121555](https://github.com/mi161374/pytorch_unet/assets/70301469/d75f8bb9-7301-45c0-835d-8996ae3f4c25)



## Contact

mahdi.imani@unimelb.edu.au
