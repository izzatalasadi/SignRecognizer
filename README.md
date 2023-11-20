# SignRecognizer
Python script implements a sign language recognition system using a combination of computer vision, machine learning, and natural language processing techniques. The system is designed to recognize signs captured through video frames, translate them into text, and suggest related words using a language model.

## Key Components:

### SignRecognizer Class:
#### Initialization:
  Loads a pre-trained GPT-2 language model and tokenizer.
  Loads the sign language dataset from CSV files (train and test sets).
  Initializes variables for sentence construction and prepares data for training.
#### Data Preparation:
  Divides the dataset into training, validation, and test sets.
  Converts labels to categorical format and normalizes image data.
#### Model Building:
  Constructs a Convolutional Neural Network (CNN) using Keras for sign language classification.
  Utilizes data augmentation techniques for improved model generalization.
  Training and Evaluation:
  Implements model training with learning rate reduction and evaluation on the test set.
  Outputs training success or error messages.
#### Video Processing:
  Utilizes OpenCV and MediaPipe for real-time hand detection and recognition in video frames.
  Displays the predicted sign and related sign image on the video feed.
  Utilizes GPT-2 to suggest words based on recognized signs.
#### Model Loading and Saving:
  Supports loading and saving of trained models.
  Suggested Words Generation:
  Uses GPT-2 to generate suggested words based on recognized signs.
#### Hand Detection:
  Utilizes the MediaPipe library for detecting hands in video frames.
  Calculates bounding boxes around detected hands for further processing.
#### Auxiliary Functions:
  Various helper functions for displaying information, handling no hand detection, and calculating bounding boxes.

## Execution:
### Dependencies
  * TensorFlow, Keras, OpenCV, MediaPipe, Matplotlib, Pandas, NumPy, and the Transformers library for GPT-2.
  * install requirements: pip install -r requirements.txt
    
### The script instantiates the SignRecognizer class and performs the following actions:
  * Loads a pre-trained model if available.
  * Optionally plots a sample of sign language images with labels.
  * Trains the model using the provided dataset.
  * Evaluates the model's performance on the test set.
  * Saves the trained model.

## Note:

  * This script uses sign language dataset in CSV format ("sign_mnist_train.csv" and "sign_mnist_test.csv"). Ensure these files are available in the specified locations.
  * Datasets can be find in this link "https://www.kaggle.com/code/arjaiswal/sign-mnist-using-cnn/input"

## Done and future work
### Done
  Build model, train the model, evaluate the model, Hand detection, sign detection, sign to letter. 
### undone 
  live translation, and words suggestion
