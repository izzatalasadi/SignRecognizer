import os
import time

import cv2
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from keras.callbacks import ReduceLROnPlateau
from keras.layers import (BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class SignRecognizer:
    def __init__(self):
        # Initialize GPT-2 language model and tokenizer
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Load sign language dataset
        self.sign_train, self.sign_test = self.load_sign_data()

        # Initialize variables for sentence construction
        self.sentence = []

        # Prepare data and build the model
        self.num_classes = 0
        self.prepare_data()
        self.model = self.build_model()

        # Initialize hand detection model
        self.hand_detection_net = mp.solutions.hands.Hands()

    def load_sign_data(self):
        # Load sign language dataset from CSV files
        sign_train = pd.read_csv("sign_mnist_train.csv")
        sign_test = pd.read_csv("sign_mnist_test.csv")
        return sign_train, sign_test

    def prepare_data(self):
        # Prepare data for training, validation, and testing
        labels = sorted(self.sign_train['label'].unique())
        self.num_classes = len(labels) + 1
        train_data, val_data = train_test_split(self.sign_train, test_size=0.2)
        self.X_train, self.y_train = self.process_data(train_data)
        self.X_val, self.y_val = self.process_data(val_data)
        self.X_test, self.y_test = self.process_data(self.sign_test)
        
    def process_data(self, data):
        # Process data and convert labels to categorical
        X = data.loc[:, data.columns != 'label'].values
        y = data.loc[:, 'label'].values
        X_reshaped = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y = to_categorical(y, num_classes=self.num_classes)
        return X_reshaped, y

    def build_model(self):
        # Build a convolutional neural network model
        model = Sequential()
        model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 512 , activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units = self.num_classes , activation = 'softmax'))
        model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
        return model

    def data_augmentation(self, x_train):
        # Apply data augmentation to the training data
        datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=False,
            vertical_flip=False
        )
        datagen.fit(x_train)
        return datagen

    def train_model(self, epochs=10, batch_size=128):
        try:
            # Implement learning rate reduction during training
            learning_rate_reduction = ReduceLROnPlateau(
                monitor='val_accuracy',
                patience=2,
                verbose=1,
                factor=0.5,
                min_lr=0.00001
            )
            datagen = self.data_augmentation(self.X_train)
            self.model.fit(
                datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
                epochs=epochs,
                validation_data=(self.X_val, self.y_val),
                callbacks=[learning_rate_reduction]
            )
            print("Model trained successfully.")
        except Exception as e:
            print(f"An error occurred during training: {e}")

    def evaluate_model(self):
        try:
            # Evaluate the trained model on the test set
            test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
            print("Test accuracy:", test_acc)
            print("Test loss:", test_loss)
        except Exception as e:
            print(f"An error occurred during evaluation: {e}")

    def save_model(self, model_filename="sign_recognizer_model.h5"):
        try:
            # Save the trained model to a file
            self.model.save(model_filename)
            print(f"Model saved as {model_filename}")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def load_model(self, model_filename="sign_recognizer_model.h5"):
        try:
            # Load a pre-trained model from a file
            if os.path.exists(model_filename):
                self.model = load_model(model_filename)
                print(f"Model loaded from {model_filename}")
            else:
                print(f"Model file {model_filename} not found. Train the model first.")
        except Exception as e:
            print(f"An error occurred while loading the model: {e}")

    def plotting(self):
        # Plot a sample of sign language images with labels
        sample_data = self.sign_train.sample(50)
        images = sample_data.drop('label', axis=1).values
        labels = sample_data['label'].values

        plt.figure(figsize=(16, 16))
        for i in range(50):
            img = images[i].reshape(28, 28)
            label = labels[i]

            fig = plt.subplot(10, 10, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'Label: {label}')
            plt.axis('off')

        plt.show()

    def video_processor(self):
        # Process video frames to recognize sign language
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")

        label_to_letter = {i: chr(65 + i) for i in range(26)}
        no_hand_timeout = 2
        last_hand_time = time.time()

        try:
            while True:
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Error: Unable to capture frame. Exiting...")
                    break

                frame, hand_rois = self.detect_hands(frame)

                if hand_rois:
                    self.process_hand_rois(hand_rois, frame, label_to_letter)
                else:
                    if time.time() - last_hand_time > no_hand_timeout:
                        self.process_no_hands()
                        last_hand_time = time.time()

                cv2.imshow('Sign Language Translation', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            pass  # print(f"An error occurred during video processing: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def process_hand_rois(self, hand_rois, frame, label_to_letter):
        suggested_words = []  # List to store suggested words for each hand

        for i, hand_roi in enumerate(hand_rois):
            gray_hand_roi = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
            resized_hand_roi = cv2.resize(gray_hand_roi, (28, 28), interpolation=cv2.INTER_AREA)
            hand_array = np.array(resized_hand_roi).astype('float32') / 255
            hand_array = hand_array.reshape(-1, 28, 28, 1)

            predicted_probabilities = self.model.predict(hand_array)
            predicted_sign = np.argmax(predicted_probabilities, axis=1)
            label = label_to_letter[predicted_sign[0]]

            sign_image = self.sign_train[self.sign_train['label'] == predicted_sign[0]].iloc[0, 1:].values

            if not np.all(np.isnan(sign_image)):
                self.display_prediction(frame, i, label, sign_image)

                # Apply sign recognition step for word suggestion
                #suggested_word = self.generate_suggested_words(label)[0]
                #suggested_words.append(suggested_word)

                # Print or use the generated sentence

        # Display suggested words on the frame
        #self.display_suggested_words(frame, suggested_words)

    def generate_suggested_words(self, word):
        # Generate suggested words using GPT-2 language model
        input_ids = self.gpt2_tokenizer.encode(word, return_tensors='pt')

        # Set attention mask to 1 for input tokens
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

        # Set pad token ID to eos_token_id
        pad_token_id = self.gpt2_tokenizer.eos_token_id

        output = self.gpt2_model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id
        )

        suggested_words = self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        suggested_words = suggested_words.split()[:5]  # Adjust the number of words as needed
        return suggested_words

    def generate_sentence(self, suggested_words):
        sentence = ' '.join(suggested_words)
        return sentence

    def display_suggested_words(self, frame, suggested_words):
        # Display suggested words on the video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        y_offset = 20

        for i, word in enumerate(suggested_words):
            cv2.putText(frame, f"Suggested word {i + 1}: {word}", (10, y_offset + i * 20),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def display_prediction(self, frame, i, label, sign_image):
        # Display predicted sign and sign image on the video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Predicted sign {i + 1}: {label}', (10, 50 + i * 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        small_image = cv2.resize(sign_image.reshape(28, 28).astype('float32') / 255, (100, 100))
        small_image = (small_image * 255).astype('uint8')
        frame[10 + i * 110:110 + i * 110, -110:-10] = cv2.cvtColor(small_image, cv2.COLOR_GRAY2RGB)

    def process_no_hands(self):
        # Handle case when no hands are detected for a while
        print("No hands detected for a while...")

    def detect_hands(self, frame):
        # Detect hands using MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hand_detection_net.process(rgb_frame)

        hand_rois = []

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                bounding_box = self.calculate_bounding_box(landmarks, frame.shape[1], frame.shape[0])

                hand_roi = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]].copy()
                hand_rois.append(hand_roi)

        return frame, hand_rois

    @staticmethod
    def calculate_bounding_box(landmarks, width, height):
        # Calculate bounding box around hand landmarks
        x_min, x_max = width, 0
        y_min, y_max = height, 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        return x_min, y_min, x_max, y_max

if __name__ == "__main__":
    # Main execution block
    recognizer = SignRecognizer()  # Create an instance of the sign recognizer

    # Commenting out the following lines to avoid training and model evaluation during GitHub commits
    # recognizer.load_model()
    # recognizer.plotting()  # Example of trained signs
    recognizer.train_model()  # Train the model using the dataset
    recognizer.evaluate_model()  # Evaluate the model's performance

    recognizer.save_model()
    
    # recognizer.load_model()
    #recognizer.video_processor()  # Predict sign in the specified frame
