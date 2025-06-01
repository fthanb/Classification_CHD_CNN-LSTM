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

## Normalizing Data (Pre-process)
* The original dataset contains 110 patients with various types of CHD, as detailed in the `imageCHD_dataset_info.xlsx` file.
* For this project, data from 72 specific patients was selected based on categories defined in a CHD paper: "Sianotik," "Non-Sianotik," and "Normal".
    * **Sianotik:** `[1008, 1010, 1012, 1015, 1028, 1037, 1046, 1050, 1064, 1074, 1085, 1092, 1099, 1105, 1111, 1113, 1120, 1125, 1129, 1141, 1145, 1146, 1147, 1150, 1158, 1178]`
    * **Non-Sianotik:** `[1001, 1002, 1007, 1011, 1014, 1018, 1019, 1020, 1025, 1029, 1033, 1035, 1036, 1041, 1047, 1061, 1070, 1079, 1103, 1109, 1132, 1133, 1135, 1139, 1140, 1148]`
    * **Normal:** `[1003, 1005, 1032, 1051, 1062, 1063, 1066, 1067, 1072, 1078, 1080, 1083, 1101, 1116, 1117, 1119, 1127, 1128, 1143, 1144]`

* The number of frames (slices) for each patient varies within the raw NIfTI files. To ensure consistency for the deep learning models, all patient scans are standardized to a uniform length of 275 frames.
    * **For patients with fewer than 275 frames:** Frames are duplicated from the middle sequence where the heart is clearly visible.
        * *Example 1:* For a patient with 206 frames, 69 data points (frames 103-172) are duplicated from the middle to reach a total of 275 frames.
        * *Example 2:* For a patient with 137 frames, the entire set of frames is duplicated and repeated to reach 275 frames.
    * **For patients with more than 275 frames:** Excess frames are removed to standardize the length. For example, if a patient has 340 frames, 65 frames (340 - 275) are removed: 33 from the beginning and 32 from the end, ensuring an even distribution of removal.
* NIfTI data is converted into PNG image format for easier processing by the deep learning frameworks.
* Subsequently, the resolution of the data is resized from 512x512 pixels to 256x256 pixels. This reduction helps in making the model training process more memory-efficient and faster.
* **Conclusion of Preprocessed Dataset:**
    * Total folders (patients): 72
    * Total image files (data): 19800

## CNN Basic (`cnn.ipynb`)

First try is a basic Convolutional Neural Network for classifying individual CT scan slices

* **Data Handling:**
    * The dataset path (`root_dir`) is set to `C:\Users\risuser\Documents\RISET_FATHAN\dataset`, which is expected to contain `ct_<patient_id>` folders with the preprocessed 275 PNG image slices.
    * **Patient-level splitting:** Patients are first divided into train 80%, validation 10%, and test sets 10% based on specific counts (Normal: 16/2/2; Sianotik: 20/3/3; Non-Sianotik: 20/3/3). This ensures that all slices from a given patient are contained within a single split (e.g., if patient 1003 is in the training set, all 275 of their slices will only be used for training).
    * A custom `CTScanDataset` loads individual PNG images, associating each with its patient's pre-assigned label (Sianotik: 0, Non-Sianotik: 1, Normal: 2).
    * **Image Augmentation:** A `transforms.Compose` pipeline is applied to the training images, including `transforms.Resize((128, 128))`, `transforms.RandomHorizontalFlip(p=0.5)`, `transforms.RandomRotation(degrees=15)`, and `transforms.ColorJitter(brightness=0.1, contrast=0.1)`. These augmentations introduce variability to the training data, helping the model generalize better.
    * **Normalization:** Images are converted to tensors and normalized to a range of [-1, 1] using `transforms.Normalize((0.5,), (0.5,))`.
    * `DataLoader`s are set up with a `batch_size` of 32 for training, validation, and testing.

* **Model Architecture (`EnhancedCNN`):**
    * **Input Layer:** Accepts individual grayscale PNG image slices of size 128x128 pixels (after resizing transform). The input shape to the model is typically $(Batch\_size, 1, 128, 128)$.
    * **Convolutional Blocks (`self.conv_layers`):** The model uses three sequential blocks for feature extraction.
        * **Block 1:** `nn.Conv2d(1, 16, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(16)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.3)`.
        * **Block 2:** `nn.Conv2d(16, 32, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(32)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.4)`.
        * **Block 3:** `nn.Conv2d(32, 64, kernel_size=3, padding=1)`, followed by `nn.BatchNorm2d(64)`, `nn.ReLU()`, `nn.MaxPool2d(2, 2)`, and `nn.Dropout(0.5)`.
    * **Flattening:** The 3D output of the last convolutional/pooling layer (which is $64 \times 16 \times 16$) is flattened into a 1D vector.
    * **Fully Connected Layers (`self.fc_layers`):** These layers perform the final classification.
        * `nn.Linear(64 * 16 * 16, 256)`
        * `nn.Dropout(0.6)`
        * `nn.ReLU()`
        * `nn.Linear(256, 128)`
        * `nn.Dropout(0.5)`
        * `nn.ReLU()`
        * `nn.Linear(128, 3)` (for 3 output classes).
    * **Forward Pass (`forward` method):** The input `x` is passed sequentially through `self.conv_layers`, then flattened, and finally passed through `self.fc_layers`.

* **Training and Evaluation:**
    * **Loss Function:** `nn.CrossEntropyLoss` is used, with `weight` calculated based on inverse class frequencies in the training set to address class imbalance.
    * **Optimizer:** `torch.optim.AdamW` is used with a learning rate of `0.00005` and `weight_decay` of `1e-4`.
    * **Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau` monitors the validation loss (`mode='min'`) and reduces the learning rate by a `factor=0.3` if no improvement is seen for `patience=3` epochs.
    * **Epochs:** The model is trained for 50 epochs.
    * **Model Saving:** The model's state dictionary is saved as `best_model.pth` whenever a new `best_val_loss` is achieved during training.
    * **Performance Tracking:** Training and validation loss and accuracy are recorded per epoch.
    * **Evaluation:** and **Test Accuracy:** are showned in the code.
