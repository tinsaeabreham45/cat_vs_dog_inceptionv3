# Cat vs. Dog Image Classification using Transfer Learning (InceptionV3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tinsaeabreham45/cat_vs_dog_inceptionv3/blob/main/Transfer_lerning.ipynb)

This repository contains a Jupyter Notebook demonstrating the basics of transfer learning for image classification. The goal is to classify images as either containing a 'cat' or a 'dog' using the pre-trained InceptionV3 model.

## Overview

The notebook covers the following key steps:

1.  **Data Acquisition:** Downloads the "Cats vs. Dogs" dataset from the Kaggle competition (via a Microsoft download link).
2.  **Data Preparation:**
    *   Extracts the dataset.
    *   Creates separate directories for training and testing sets.
    *   Splits the data (90% training, 10% testing), handling potential zero-byte images.
3.  **Data Augmentation:** Uses `ImageDataGenerator` from Keras to apply real-time data augmentation (rotation, shifts, shear, zoom, flip) to the training images to improve model robustness. Validation data is only rescaled.
4.  **Transfer Learning Model:**
    *   Loads the **InceptionV3** model pre-trained on ImageNet, excluding the final classification layer (`include_top=False`).
    *   Loads the pre-trained weights.
    *   Freezes the layers of the base InceptionV3 model to retain the learned features.
    *   Adds custom classification layers on top (Flatten, Dense ReLU, Dense Sigmoid).
5.  **Model Compilation:** Compiles the model using the RMSprop optimizer with a low learning rate and binary cross-entropy loss function.
6.  **Model Training:** Trains the model on the augmented training data and evaluates it on the validation data for a small number of epochs (2 in this notebook).
7.  **Prediction:** Includes a section to upload a custom image and use the trained model to predict whether it's a cat or a dog.
8.  **Model Saving:** Saves the trained model to an HDF5 file (`.h5`).

## Dataset

*   **Source:** Kaggle Cats and Dogs Dataset
*   **Download Link Used:** `https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip`
*   **Size:** Approximately 25,000 images (12,501 cats, 12,501 dogs).
*   **Split:** 90% Training / 10% Testing.

## Requirements

*   Python 3
*   TensorFlow / Keras
*   NumPy
*   Pillow (implicitly used by Keras image functions)
*   Matplotlib (Optional, for plotting training history if added)
*   A computing environment like Google Colab (recommended, especially with GPU acceleration) or a local machine with sufficient resources.

## Usage

1.  **Open in Colab:** Click the "Open In Colab" badge at the top of this README.
2.  **Select Runtime:** In Colab, go to `Runtime` -> `Change runtime type` and select `GPU` as the Hardware accelerator for faster training.
3.  **Run Cells:** Execute the notebook cells sequentially from top to bottom.
    *   The notebook will automatically download the dataset and weights.
    *   It will prepare the data directories.
    *   It will define, compile, and train the model.
    *   *(Note: The prediction cell requires user interaction to upload a file. Ensure previous cells, especially model definition and training, have run successfully before executing the prediction cell.)*
4.  **Prediction:** Run the final cells to upload your own cat or dog image and see the model's prediction.
5.  **Download Model:** The last cell saves the trained model and provides a download link (specific to the Colab environment).

## Results

The transfer learning approach achieves high validation accuracy (around 96%) relatively quickly (within 2 epochs), demonstrating the power of leveraging pre-trained models for image classification tasks.

## Model File

The notebook saves the trained model weights and architecture in the HDF5 format (`.h5`). Note that Keras recommends using the newer `.keras` format for saving models.
