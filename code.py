from sklearn.svm import SVC  # Support Vector Classifier from scikit-learn
from sklearn.linear_model import LogisticRegression  # Logistic Regression from scikit-learn
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors from scikit-learn
from sklearn.preprocessing import StandardScaler  # Standard scaler for feature scaling
from tensorflow.keras.models import Sequential  # Sequential model from TensorFlow
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout  # Layers for CNN
from tensorflow.keras.optimizers import Adam  # Adam optimizer from TensorFlow
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Image data generator for preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Metrics for evaluation
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Data visualization library
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical computations

class CancerClassifier:
    def __init__(self):
        # Initialize class variables for class names and the number of classes
        self.class_names = ['benign', 'malignant']
        self.num_classes = len(self.class_names)
        self.load_data()  # Load and preprocess the dataset

    def load_data(self):
        """Load and preprocess the skin cancer dataset."""
        # Define directories for training and testing data
        train_dir = 'archive/train'
        test_dir = 'archive/test'

        # Initialize an image data generator to rescale pixel values to [0, 1]
        data_generator = ImageDataGenerator(rescale=1/255.0)

        # Load and preprocess training data
        train_data = data_generator.flow_from_directory(
            train_dir,
            target_size=(64, 64),  # Resize images to 64x64
            class_mode='sparse',  # Sparse labels for multi-class classification (benign/malignant)
            shuffle=True,  # Shuffle the data for better training 
            seed=1  # Set seed for reproducibility 
        )
        # Combine batches into a single array for training data

        # Combine all feature batches into a single array using vertical stacking (row-wise),
        # resulting in a unified dataset of training features.
        self.X_train = np.vstack([train_data[i][0] for i in range(len(train_data))])

        # Combine all label batches into a single array using horizontal stacking (flattening),
        # resulting in a unified dataset of training labels.
        self.y_train = np.hstack([train_data[i][1] for i in range(len(train_data))])

        # Load and preprocess testing data
        test_data = data_generator.flow_from_directory(
            test_dir,
            target_size=(64, 64),  # Resize images to 64x64
            class_mode='sparse',  # Sparse labels for multi-class classification
            shuffle=False,  # Do not shuffle test data
            seed=1  # Set seed for reproducibility
        )
        # Combine all feature batches from the test data into a single 4D array using vertical stacking (row-wise),
        # resulting in a unified dataset of testing features.
        self.X_test = np.vstack([test_data[i][0] for i in range(len(test_data))])

        # Combine all label batches from the test data into a single 1D array using horizontal stacking (flattening),
        # resulting in a unified dataset of testing labels.
        self.y_test = np.hstack([test_data[i][1] for i in range(len(test_data))])


    def enhance_contrast(self, image):
        """Apply histogram equalization to improve image contrast."""
        # Convert normalized image to 8-bit integer values
        image_8bit = (image * 255).astype('uint8')
        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(image_8bit)
        # Return the normalized equalized image
        return equalized / 255.0

    def detect_edges(self, image):
        """Apply edge detection using Sobel filters to highlight edges."""
        # Define Sobel kernels for x and y gradients
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Initialize arrays for storing edge gradients
        edges_x = np.zeros_like(image)
        edges_y = np.zeros_like(image)

        # Convolve kernels with the image to calculate gradients
        for i in range(1, image.shape[0] - 1):
            for j in range(1, image.shape[1] - 1):
                edges_x[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel_x)
                edges_y[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * kernel_y)

        # Combine gradients to compute edge magnitude
        return np.sqrt(edges_x**2 + edges_y**2)

    def extract_features(self, image):
        """Extract basic features from an image."""
        features = []

        # Calculate average intensity of the image
        features.append(np.mean(image))

        # Calculate the standard deviation of intensity
        features.append(np.std(image))

        # Calculate average edge intensity using Sobel filters
        edges = self.detect_edges(image)
        features.append(np.mean(edges))

        # Calculate quadrant intensities
        h, w = image.shape
        features.append(np.mean(image[:h//2, :w//2]))  # Top-left
        features.append(np.mean(image[:h//2, w//2:]))  # Top-right
        features.append(np.mean(image[h//2:, :w//2]))  # Bottom-left
        features.append(np.mean(image[h//2:, w//2:]))  # Bottom-right

        # Return extracted features as a NumPy array
        return np.array(features)

    def evaluate_model(self, y_true, y_pred, model_name):
        """Evaluate model performance and display metrics."""
        print(f"\n{model_name} Results:")
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def run_experiments(self):
        """Run experiments with Logistic Regression, KNN, and SVM."""

        # Flatten the training images from 4D (num_samples, height, width, channels)
        # to 2D (num_samples, num_features), where num_features is the total number 
        # of pixels per image (height * width * channels).
        X_train_flattened = self.X_train.reshape(self.X_train.shape[0], -1)

        # Flatten the testing images from 4D (num_samples, height, width, channels)
        # to 2D (num_samples, num_features) for compatibility with models that 
        # require 2D input data.
        X_test_flattened = self.X_test.reshape(self.X_test.shape[0], -1)


        # Logistic Regression
        print("\nRunning Logistic Regression...")
        # Initialize the Logistic Regression model with a maximum of 1000 iterations for training to find the best parameters (weights) for the logistic regression model.
        log_reg = LogisticRegression(max_iter=1000)

        # Train the Logistic Regression model on the flattened training data (X_train_flattened) and corresponding labels (y_train)
        log_reg.fit(X_train_flattened, self.y_train)

        # Use the trained model to predict the labels of the flattened test data (X_test_flattened)
        log_reg_predictions = log_reg.predict(X_test_flattened)

        # Evaluate the model's performance by comparing the true test labels (y_test) with the predicted labels (log_reg_predictions)
        # The evaluation results are displayed with the label "Logistic Regression"
        self.evaluate_model(self.y_test, log_reg_predictions, "Logistic Regression")

        # K-Nearest Neighbors (KNN)
        print("\nRunning K-Nearest Neighbors (KNN)...")
        knn = KNeighborsClassifier(n_neighbors=3)  # Initialize KNN with 3 neighbors
        knn.fit(X_train_flattened, self.y_train)  # Train KNN on flattened data
        knn_predictions = knn.predict(X_test_flattened)  # Predict test labels
        self.evaluate_model(self.y_test, knn_predictions, "KNN")

        # Support Vector Machine (SVM)
        print("\nRunning Support Vector Machine (SVM)...")
        # Initialize the Support Vector Machine (SVM) model with the following parameters:
        # - kernel='linear': Using a linear kernel to find a linear decision boundary
        # - C=1.0: Regularization parameter that controls the trade-off between training error and model complexity
        # - gamma='scale': Specifies the kernel coefficient, automatically calculated based on input features
        svm = SupportVectorMachine(kernel='linear', C=1.0, gamma='scale')

        # Train the SVM model on the flattened training data (X_train_flattened) and the true labels (y_train)
        svm.fit(X_train_flattened, self.y_train)

        # Use the trained SVM model to predict the labels for the flattened test data (X_test_flattened)
        svm_predictions = svm.predict(X_test_flattened)

        # Evaluate the model's performance by comparing the true test labels (y_test) with the predicted labels (svm_predictions)
        # The evaluation results are displayed with the label "SVM"
        self.evaluate_model(self.y_test, svm_predictions, "SVM")

    def build_neural_network(self):
        """Build and train the neural network."""
        print("Building and training neural network...")

        # Build the model
        # Define a Sequential model where each layer is connected to the next
        model = Sequential([
            # Flatten the input images (64x64 RGB) into a 1D array of 12,288 elements
            Flatten(input_shape=(64, 64, 3)),  # Input shape of 64x64x3 (RGB images) is flattened to 1D
            
            # Fully connected layer with 128 neurons and ReLU activation
            # ReLU introduces non-linearity, enabling the model to learn complex patterns
            Dense(128, activation='relu'),  
            
            # Dropout layer to prevent overfitting by randomly dropping 50% of neurons during training
            Dropout(0.5),  
            
            # Output layer with 'self.num_classes' neurons, each representing one class
            # Softmax activation function outputs a probability distribution across all classes
            Dense(self.num_classes, activation='softmax')  
        ])


        # Compile the model with the specified optimizer, loss function, and evaluation metric
        model.compile(
            optimizer='adam',  # Adam optimizer for efficient and adaptive training
            loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
            metrics=['accuracy']  # Metric to track model performance (accuracy)
        )

        # Train the model using the training data (X_train and y_train)
        # Specify the number of epochs, batch size, and validation data
        history = model.fit(
            self.X_train,  # Input features for training
            self.y_train,  # True labels for training
            epochs=10,  # Train for 10 epochs (complete passes through the training data)
            batch_size=32,  # Process 32 samples at a time before updating weights
            validation_split=0.2,  # Reserve 20% of the data for validation after each epoch
            verbose=1  # Display progress bar and training statistics during training
        )


        # Plot training history
        self.plot_training_history(history, "Neural Network Training History")

        # Evaluate the model on the test data
        _, test_accuracy = model.evaluate(self.X_test, self.y_test)

        # Make predictions
        y_pred = np.argmax(model.predict(self.X_test), axis=1)

        # Use evaluate_model to visualize results
        self.evaluate_model(self.y_test, y_pred, "Neural Network")

        return history, test_accuracy

    def run_cnn(self):
        """Run a Convolutional Neural Network (CNN)."""
        # Print a message indicating the start of CNN training
        print("\nRunning Convolutional Neural Network...")

        # Define the CNN architecture using the Sequential model
        model = Sequential([
            # First convolutional layer: 32 filters of size 3x3 with ReLU activation, expecting 64x64x3 images
            Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
            
            # Max pooling layer: 2x2 pool size to reduce spatial dimensions
            MaxPooling2D(pool_size=(2, 2)),
            
            # Dropout layer: randomly drop 25% of neurons to prevent overfitting
            Dropout(0.25),
            
            # Second convolutional layer: 64 filters of size 3x3 with ReLU activation
            Conv2D(64, (3, 3), activation='relu'),
            
            # Max pooling layer: 2x2 pool size to reduce spatial dimensions
            MaxPooling2D(pool_size=(2, 2)),
            
            # Dropout layer: randomly drop 25% of neurons to prevent overfitting
            Dropout(0.25),
            
            # Flatten the 2D output from convolutional layers to 1D for input to dense layers
            Flatten(),
            
            # Fully connected layer with 128 neurons and ReLU activation
            Dense(128, activation='relu'),
            
            # Dropout layer: randomly drop 50% of neurons to prevent overfitting
            Dropout(0.5),
            
            # Output layer: number of neurons equals number of classes (len(self.class_names))
            # Softmax activation to output probabilities for each class
            Dense(len(self.class_names), activation='softmax')
        ])

        # Compile the model with Adam optimizer, sparse categorical cross-entropy loss, and accuracy as the evaluation metric
        model.compile(
            optimizer=Adam(learning_rate=0.001),  # Adam optimizer with learning rate 0.001
            loss='sparse_categorical_crossentropy',  # Loss function for multi-class classification with integer labels
            metrics=['accuracy']  # Metric to evaluate the model (accuracy)
        )

        # Train the model on the training data with validation split
        history = model.fit(
            self.X_train,  # Training input data
            self.y_train,  # Training labels
            epochs=10,  # Number of training epochs
            batch_size=32,  # Number of samples per batch
            validation_split=0.2,  # Split 20% of data for validation during training
            verbose=1  # Display training progress (with progress bar)
        )


        # Plot training history
        self.plot_training_history(history, "CNN Training History")

        # Make predictions
        predictions = np.argmax(model.predict(self.X_test), axis=1)

        # Evaluate the model
        self.evaluate_model(self.y_test, predictions, "Convolutional Neural Network")

        return history
    
    def plot_training_history(self, history, title="Model Training History"):
        """Plot training and validation metrics over epochs."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training and validation loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Plot training and validation accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

class SupportVectorMachine:
    def __init__(self, kernel='linear', C=1.0, gamma='scale'):
        """Initialize the Support Vector Machine (SVM) with given parameters."""
        # Initialize SVM parameters: kernel, regularization parameter (C), and gamma for RBF kernel
        self.kernel = kernel  # The type of kernel ('linear', 'rbf', etc.)
        self.C = C  # Regularization parameter to control the trade-off between fitting the data and model complexity
        self.gamma = gamma  # Parameter for non-linear kernel function (RBF, etc.)

    def fit(self, X, y):
        """Fit the SVM model on the training data."""
        self.scaler = StandardScaler()  # Initialize a standard scaler for feature scaling

        # Flatten the images (or data) to 2D (num_samples, num_features) for SVM processing
        num_samples = X.shape[0]  # Number of samples (images or data points)
        X_flattened = X.reshape(num_samples, -1)  # Flatten 3D images (height, width, channels) to 2D

        # Scale the features to have zero mean and unit variance
        X_scaled = self.scaler.fit_transform(X_flattened)  # Apply scaling to the flattened data

        # Initialize the Support Vector Classifier with the given kernel, C, and gamma values
        self.model = SVC(kernel=self.kernel, C=self.C, gamma=self.gamma)  # Initialize SVM model

        # Train the model with scaled features and corresponding labels
        self.model.fit(X_scaled, y)  # Fit the model using the scaled training data

    def predict(self, X):
        """Make predictions using the trained SVM model."""
        # Flatten the input data (images) to 2D for prediction
        num_samples = X.shape[0]  # Number of samples to predict
        X_flattened = X.reshape(num_samples, -1)  # Flatten 3D images (height, width, channels) to 2D

        # Scale the test data using the previously fitted scaler
        X_scaled = self.scaler.transform(X_flattened)  # Apply scaling to the test data

        # Use the trained SVM model to predict the labels for the test data
        return self.model.predict(X_scaled)  # Return the predicted labels

if __name__ == "__main__":
    # Initialize the classifier
    classifier = CancerClassifier()

    # Load the dataset
    classifier.load_data()

    # Run classification experiments with Logistic Regression, KNN, and SVM
    classifier.run_experiments()

    # Run experiments with a neural network
    classifier.build_neural_network()

    # Run experiments with a CNN
    classifier.run_cnn()
