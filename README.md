# GPOSC-Net

## Description  
GPOSC-Net is a generative prediction model for orthognathic surgery outcomes. This model synthesizes **post-operative lateral cephalograms** from pre-operative data, providing precise predictions. GPOSC-Net consists of **a landmark prediction model** that estimates post-surgical cephalometric changes and **a latent diffusion model** that generates realistic post-operative cephalogram images based on predicted landmarks and segmented profile lines.  

Through validation using diverse patient datasets, a visual Turing test, and simulation studies, GPOSC-Net has demonstrated high accuracy in cephalometric landmark prediction and high-fidelity image synthesis. This tool enhances **clinical decision-making and patient communication** by improving predictive accuracy and visualization.  

## Installation  
GPOSC-Net follows the official **Latent Diffusion** installation method. To run this project, ensure that your environment meets the requirements specified in the **Latent Diffusion official documentation** and install the necessary dependencies accordingly.  

## Usage  
To run GPOSC-Net, use the following commands:  

### Inference & Prediction 
```bash
python main.py
```
Generates predicted post-operative lateral cephalograms based on pre-operative data.
The resulting images are saved in the specified output directory.

## Features & Directory Structure
* configs/inference.yaml - Configuration file for model, data, training, and inference
* data/ - Directory containing sample datasets
* ldm/ - Codebase for model training and inference
* main.py - Script for training the model
* inference.py - Script for model inference

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute it.
