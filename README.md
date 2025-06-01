# Classification of CHD using CNN+LSTM

##  Project Overview

This project aims to leverage deep learning models for the classification of Congenital Heart Disease (CHD) using 3D Computed Tomography (CT) scan data. The primary objective is to accurately categorize patient scans into distinct classes: Cyanotic, Non-Cyanotic, and Normal(non-CHD). The research explores deep learning architectures to understand their effectiveness in this medical imaging task. This includes an **Convolutional Neural Network (CNN)** for 2D image slice classification, as well as **hybrid CNN-Long Short-Term Memory (LSTM) models** designed to process the inherent sequential information within 3D CT scan volumes. **K-Fold Cross-Validation** strategy is also implemented

## Tools & Requirements

* **Python 3.11**
* **PyTorch 2.5.1** 
* **NumPy** 
* **Pillow (PIL)** 
* **Nibabel** 
* **Scikit-learn**
* **Matplotlib & Seaborn** 
* **tqdm** 

## Data Source
The dataset used in this project is from Kaggle:
[https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd](https://www.kaggle.com/datasets/xiaoweixumedicalai/imagechd).

To extract the dataset files:
1. Download the zipped file from the Kaggle link.
2. You might find the downloaded file has a non-standard extension (e.g., ".change2zip"). **Rename this file's extension to ".zip"** (e.g., `ImageCHD_dataset.change2zip` becomes `ImageCHD_dataset.zip`).
3. **Extract all files** from the renamed `.zip` archive.

