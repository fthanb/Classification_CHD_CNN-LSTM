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
