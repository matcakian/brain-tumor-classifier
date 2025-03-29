# Brain Tumor Classifier

Classifies brain MRI scans into four categories using a convolutional neural network in TensorFlow. Uses a dataset from kaggle.com (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) for training.


## Setup

Install dependencies:

```zsh
pip3 install tensorflow opencv-python numpy scikit-learn
```


## Usage


1. Prepare dataset

```zsh
python3 prepare_dataset.py
```
This generates train.npz and test.npz files from your dataset folders.

2. Train the model

```zsh
python3 scripts/train_cnn_model.py
```
Trains the model and saves it as cnn_model.h5


3. Run a prediction

```zsh
python3 scripts/client.py path/to/model.h5 path/to/image.jpg
```
Returns the predicted tumor class and confidence.

