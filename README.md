# Skin Cancer Classification Project

This project involves building a skin cancer classification system using machine learning and deep learning techniques. The system identifies whether a given skin lesion is benign or malignant, leveraging image processing, feature extraction, and multiple classification algorithms.

## Project Overview

The primary steps in this project are:

1. **Data Loading and Preprocessing**:
    - The dataset is loaded using TensorFlow's `ImageDataGenerator`.
    - Images are rescaled to normalize pixel values and processed into training and testing sets.

2. **Feature Extraction**:
    - Techniques like histogram equalization and edge detection are applied to improve image quality.
    - Basic image features, such as average intensity, standard deviation, and quadrant-based intensities, are extracted.

3. **Model Training and Evaluation**:
    - Several models are implemented:
        - Logistic Regression
        - K-Nearest Neighbors (KNN)
        - Support Vector Machine (SVM)
        - Fully Connected Neural Networks (FCNN)
        - Convolutional Neural Networks (CNN)
    - Evaluation metrics include accuracy, classification reports, and confusion matrices.

4. **Visualization**:
    - Performance metrics and confusion matrices are visualized using Matplotlib and Seaborn.

## Key Components

### Class: `CancerClassifier`

- **Methods**:
    - `load_data`: Loads and preprocesses the dataset.
    - `enhance_contrast`: Applies histogram equalization to enhance image contrast.
    - `detect_edges`: Detects edges in the image using Sobel filters.
    - `extract_features`: Extracts various features from the image for classification.
    - `evaluate_model`: Computes metrics and visualizes results.
    - `run_experiments`: Runs Logistic Regression, KNN, and SVM experiments.
    - `build_neural_network`: Builds and trains a fully connected neural network.
    - `run_cnn`: Builds and trains a convolutional neural network.

### Class: `SupportVectorMachine`

- Implements SVM with support for kernel selection, feature scaling, and classification.

## Prerequisites

- Python 3.8+
- Libraries:
    - `numpy`
    - `matplotlib`
    - `seaborn`
    - `scikit-learn`
    - `tensorflow`
    - `opencv-python`

Install dependencies using:
```bash
pip install numpy matplotlib seaborn scikit-learn tensorflow opencv-python
```

## How to Run

1. Ensure the dataset is structured with `train` and `test` folders under `archive/`.
2. Execute the script:
    ```bash
    python <script_name>.py
    ```
3. The script will:
    - Load and preprocess the data.
    - Run experiments using Logistic Regression, KNN, SVM, FCNN, and CNN.
    - Display performance metrics and visualizations.

## Results and Metrics

The project evaluates models on:

- **Accuracy**: Overall correctness of predictions.
- **Precision, Recall, and F1-Score**: Performance for individual classes.
- **Confusion Matrix**: Visual comparison of predicted and actual labels.

The CNN typically outperforms other models due to its ability to learn hierarchical features directly from the images.

## Future Enhancements

- Add more advanced augmentation techniques to improve model robustness.
- Experiment with transfer learning using pre-trained models like ResNet or VGG.
- Optimize hyperparameters using grid or random search.
- Deploy the trained model as a web application for real-world usage.

## License

This project is open-source and available under the MIT License. Feel free to use, modify, and distribute it as needed.

