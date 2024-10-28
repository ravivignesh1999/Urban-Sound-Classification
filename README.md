# Urban Sound Classification

This project aims to classify various urban sound events using machine learning and deep learning techniques. Urban Sound Classification has multiple real-world applications, including noise monitoring, smart city implementations, and enhancing security systems.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
Urban sound classification is a machine learning problem that involves the identification of sound events from audio data collected in urban environments. This project leverages deep learning techniques to classify sounds such as car horns, sirens, street music, and more.

## Dataset
The project uses the **UrbanSound8K** dataset, which consists of 8732 labeled sound excerpts (<=4s) of urban sounds from 10 classes:
- Air Conditioner
- Car Horn
- Children Playing
- Dog Bark
- Drilling
- Engine Idling
- Gunshot
- Jackhammer
- Siren
- Street Music

Download the dataset [here](https://urbansounddataset.weebly.com/urbansound8k.html).

## Project Structure
```
Urban-Sound-Classification/
│
├── data/                  # Contains audio dataset
├── notebooks/             # Jupyter notebooks for data exploration and training
├── models/                # Trained models and model architecture files
├── src/                   # Source code for data preprocessing, model building, and evaluation
│   └── cnnv7_yamnet.py
├── results/               # Generated outputs and model performance metrics
└── README.md              # Project documentation
```

## Requirements
- Python 3.11.7
- TensorFlow 2.16.1
- Keras 3.3.3
- Librosa 0.10.0
- Numpy 1.26.4
- Pandas 2.1.4
- Matplotlib 3.7.0
- Scipy 1.11.4 

## Results
The trained model achieves an accuracy of **98%** on the test dataset.
![Confusion Matrix_Yamnet](https://github.com/user-attachments/assets/57abcce9-fca3-4ab7-8aa2-6c7bc59cdfde)


## Conclusion
The Urban Sound Classification project demonstrates the effectiveness of deep learning techniques for classifying sound events in urban environments. Potential future improvements include:
- Fine-tuning hyperparameters for better performance
- Using more advanced architectures like Recurrent Neural Networks (RNNs) for sequence learning
- Implementing additional data augmentation techniques to enhance robustness
- Employing advanced feature representation techniques are chroma stft and mel spectrograms
