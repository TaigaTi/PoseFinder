# PoseFinder

PoseFinder is a machine learning pipeline for classifying human poses (e.g., "sitting" or "standing") in images using pose estimation and supervised learning. It leverages MediaPipe for extracting pose keypoints and applies a Random Forest classifier to categorize the detected pose.

## Features

- **Keypoint Extraction**: Uses MediaPipe to extract pose keypoints (x, y coordinates) from images.
- **Dataset Builder**: Constructs a labeled dataset of pose keypoints from a directory of categorized images.
- **Model Training**: Trains a Random Forest classifier to distinguish between "sitting" and "standing" poses.
- **Prediction**: Loads an image and predicts the pose using the trained model.
- **Export Utilities**: Outputs classification results and evaluation reports to Excel.

## Repository Structure

```
.
├── build_dataset.py         # Script to build a dataset from images in 'data/' folder
├── data/                   # Directory for training images, organized by category 
├── export_data.py          # Utility to export classification reports to Excel
├── extract_keypoints.py    # Contains the keypoint extraction logic using MediaPipe
├── model/                  # Directory for trained model artifacts
├── pose_dataset.csv        # Generated dataset of keypoints and labels
├── pose_finder.py          # Main script to predict pose from an input image
├── requirements.txt        # Python dependencies
├── results/                # Directory for results, reports, and evaluation outputs
├── test.jpeg               # Example image for testing
├── train_classifier.py     # Script to train the pose classification model
└── ...
```

**Note:** Only a subset of files are listed here. For the full list, [see the repository contents](https://github.com/TaigaTi/PoseFinder/tree/main).

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/TaigaTi/PoseFinder.git
    cd PoseFinder
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Prepare Dataset

Organize your images in the `data/` directory as follows:

```
data/
├── sitting/
│   ├── image1.jpg
│   └── ...
└── standing/
    ├── image2.jpg
    └── ...
```

### 2. Build the Dataset

Extract keypoints from images and create a labeled dataset:

```bash
python build_dataset.py
```

This generates `pose_dataset.csv`.

### 3. Train the Classifier

Train a Random Forest classifier on the dataset:

```bash
python train_classifier.py
```

- Saves the trained model to `model/rf_pose_classifier.joblib`
- Outputs evaluation metrics and exports a report to `results/results.xlsx`

### 4. Predict Poses

Classify the pose in a test image (default: `test.jpeg`):

```bash
python pose_finder.py
```

- Prints whether the detected pose is "Sitting" or "Standing".

## Key Components

- **extract_keypoints.py**: Uses MediaPipe to extract pose landmarks for each image.
- **build_dataset.py**: Loops through labeled images, extracts keypoints, and saves them with labels.
- **train_classifier.py**: Trains a Random Forest model and saves it for inference.
- **pose_finder.py**: Loads an image, extracts keypoints, and predicts the pose using the trained model.
- **export_data.py**: Exports classification reports to Excel for easy analysis.