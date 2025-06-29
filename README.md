# Pet Classifier

This repository contains a binary image classification model that distinguishes between cats and dogs. The models are implemented in PyTorch and TensorFlow, each using a custom CNN architecture.

## Overview

The pet classifier works by training a convolutional neural network (CNN) on labeled images of cats and dogs, learning visual features that distinguish the two. The pipeline includes:

- Image preprocessing and augmentation
- CNN-based architecture for feature extraction
- Model training using the Cats vs. Dogs dataset
  
### Key Features:
- Uses the [Microsoft Cats vs. Dogs dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data)
- Supports both PyTorch and TensorFlow implementations (each in its own .py script)
- GPU acceleration (using Colab)
- Training, validation, and test split with accuracy and loss tracking
- Early stopping

### Accuracy
The pytorch-based model achieves ~89–90% accuracy on the test set after training for 30 epochs on a custom CNN.

## Dataset

To train or test the model, download the dataset from Kaggle:

1. Visit the [Microsoft Cats vs. Dogs dataset](https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset/data) page.
2. Download the dataset as a `.zip` file and extract the contents.
3. Place the contents into the `dataset/original` folder.
