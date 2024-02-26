import time
import os
import logging
import concurrent.futures
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from keras.layers import (BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from contextlib import contextmanager

# Configure logging
logging.basicConfig(filename='sign_recognizer.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataPreparation():
    def __init__(self, train_data="sign_mnist_train.csv", test_data="sign_mnist_test.csv") -> None:
        self.sign_train = pd.read_csv(train_data)
        self.sign_test = pd.read_csv(test_data)
        self.num_classes = len(sorted(self.sign_train['label'].unique())) + 1
        self.X_train, self.y_train = None , None
        self.X_val, self.y_val = None, None
        self.X_test, self.y_test = None, None
    
    def process_data(self, data):
        X = data.loc[:, data.columns != 'label'].values
        y = data.loc[:, 'label'].values
        X_reshaped = X.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        y = to_categorical(y, num_classes=self.num_classes)
        return (X_reshaped, y)
    
    def prepare_data(self):
        train_data, val_data = train_test_split(self.sign_train, test_size=0.2)
        self.X_train, self.y_train = self.process_data(train_data)
        self.X_val, self.y_val = self.process_data(val_data)
        self.X_test, self.y_test = self.process_data(self.sign_test)
    
class ModelCreation(DataPreparation):
    def __init__(self, model_filename="sign_recognizer_model.keras", auto_run=True, epochs=10, batch_size=128):
        super().__init__()
        
        self.model_filename = model_filename
        self.model = self.build_model()
        
        if auto_run:
            self.prepare_data()
            self.train_model(epochs=epochs, batch_size=batch_size)
            self.evaluate_model()
            self.save_model()

    def build_model(self):
        model = Sequential([
            Conv2D(75, (3,3), strides=1, padding='same', activation='relu', input_shape=(28,28,1)),
            BatchNormalization(),
            MaxPooling2D((2,2), strides=2, padding='same'),
            Conv2D(50, (3,3), strides=1, padding='same', activation='relu'),
            Dropout(0.2),
            BatchNormalization(),
            MaxPooling2D((2,2), strides=2, padding='same'),
            Conv2D(25, (3,3), strides=1, padding='same', activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2,2), strides=2, padding='same'),
            Flatten(),
            Dense(units=512, activation='relu'),
            Dropout(0.3),
            Dense(units=self.num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def data_augmentation(self):
        return ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            width_shift_range=0.1,
            height_shift_range=0.1
        )

    def train_model(self, epochs=10, batch_size=128, patience=2, verbose=1, factor=0.5, min_lr=0.00001):
        learning_rate_reduction = ReduceLROnPlateau(
            monitor='val_accuracy',
            patience=patience,
            verbose=verbose,
            factor=factor,
            min_lr=min_lr
        )
        datagen = self.data_augmentation()
        datagen.fit(self.X_train)
        self.model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(self.X_val, self.y_val),
            callbacks=[learning_rate_reduction]
        )
        logging.info("Model trained successfully.")

    def evaluate_model(self):
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        logging.info(f"Test accuracy: {test_acc}")
        logging.info(f"Test loss: {test_loss}")

    def save_model(self):
        self.model.save(self.model_filename)
        logging.info(f"Model saved in {os.getcwd()}/{self.model_filename}")

class SignRecognizer:
    def __init__(self):
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        
        model_creation = ModelCreation(auto_run=False)
        self.sign_train = model_creation.sign_train
        self.sign_test = model_creation.sign_test
        self.prev_fingertip_position = None
        self.stable_position_timer = None
        self.stable_position_duration = 0.5
        self.words = []
        self.logged_in = False  # Initialize login status

    def model_check(self, model_filename="sign_recognizer_model.keras"):
        try:
            if os.path.exists(model_filename):
                self.model = load_model(model_filename)
                logging.info(f"Model '{model_filename}' been loaded")
            else:
                logging.warning(f"Model file {model_filename} not found. Training model in progress ...")
                ModelCreation()
                logging.info(f"Model '{model_filename}' been created")
                self.model = load_model(model_filename)
                logging.info(f"Model '{model_filename}' been loaded")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")

    @contextmanager
    def video_capture_context_manager(self, index=0):
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise ValueError("Unable to open video source")
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def capture_frames(self):
        while True:
            _, frame = self.cap.read()
            if frame is None:
                logging.error("Error: Unable to capture frame. Exiting...")
                break
            yield frame
    
    def check_stable_fingertip_position(self, fingertip_position):
        if self.prev_fingertip_position is None:
            self.prev_fingertip_position = fingertip_position
            return False

        if fingertip_position == self.prev_fingertip_position:
            if self.stable_position_timer is None:
                self.stable_position_timer = time.time()
            else:
                if time.time() - self.stable_position_timer >= self.stable_position_duration:
                    return True
        else:
            self.prev_fingertip_position = fingertip_position
            self.stable_position_timer = None

        return False

    def get_fingertip_position(self, frame):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False,
                                        max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(gray_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                index_fingertip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                width, height, _ = frame.shape
                x, y = int(index_fingertip.x * width), int(index_fingertip.y * height)
                return x, y

    def pick_letter(self, frame, suggested_words):
        unique_suggested_words = list(set(str(word) for word in suggested_words))

        for i, word in enumerate(unique_suggested_words):
            cv2.putText(frame, f"Suggested word {i + 1}: {word}", (10, 20 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        while True:
            fingertip_position = self.get_fingertip_position(frame)
            if fingertip_position:
                x, y = fingertip_position
                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

                for i, word in enumerate(unique_suggested_words):
                    if 20 + i * 20 <= y <= 20 + (i + 1) * 20:
                        if self.check_stable_fingertip_position(fingertip_position):
                            logging.info("Selected word: %s", word)
                            self.words.append(word)
                            break

            cv2.imshow('Hand Tracking', frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    def video_processor(self):
        try:
            with self.video_capture_context_manager(0) as cap:
                self.cap = cap
                label_to_letter = {i: chr(65 + i) for i in range(26)}
                suggested_words = []
                pause_processing = False
                frame = None
                words = []

                while True:
                    if not pause_processing:
                        frame = cap.read()[1]

                        if frame is None or frame.size == 0:
                            continue

                        hand_rois = self.detect_hands(frame)

                        if hand_rois:
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                future = executor.submit(self.process_hand_rois, hand_rois, frame, label_to_letter)
                                result = future.result()
                                if result not in suggested_words:
                                    suggested_words.append(result)

                        logging.info("Suggested Sentence: %s", suggested_words)

                    cv2.imshow('Sign Language Translation', frame)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('c'):
                        pause_processing = not pause_processing
                        if pause_processing:
                            logging.info("Processing paused...")
                            word = self.pick_letter(frame, suggested_words)
                            words.append(word)
                            suggested_words.clear()
                        else:
                            logging.info("Processing resumed...")

        except IOError as e:
            logging.error(f"IOError occurred: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            cv2.destroyAllWindows()

    def detect_hands(self, frame):
        hands = mp.solutions.hands.Hands(static_image_mode=False,
                                        max_num_hands=1,
                                        min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(gray_frame)

        hand_rois = []

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                bounding_box = self.calculate_bounding_box(landmarks, frame.shape[1], frame.shape[0])
                hand_roi = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]].copy()
                hand_rois.append(hand_roi)

        return hand_rois
    
    @staticmethod
    def calculate_bounding_box(landmarks, width, height):
        x_min, x_max = width, 0
        y_min, y_max = height, 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            x_max = max(x_max, x)
            y_min = min(y_min, y)
            y_max = max(y_max, y)

        return x_min, y_min, x_max, y_max

    def process_hand_rois(self, hand_rois, frame, label_to_letter):
        suggested_labels = []
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
                self.display_prediction(frame, i ,label, sign_image)
                suggested_labels.append(label)
        
        return suggested_labels  

    def generate_suggested_words(self, word):
        input_ids = self.gpt2_tokenizer.encode(word, return_tensors='pt')
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
        pad_token_id = self.gpt2_tokenizer.eos_token_id

        output = self.gpt2_model.generate(
            input_ids,
            max_length=50,
            num_return_sequences=1,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id
        )

        suggested_words = self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
        suggested_words = suggested_words.split()
        return suggested_words

    def generate_sentence(self, suggested_words):
        return ' '.join(suggested_words)
        
    def display_suggested_words(self, frame, suggested_words):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        y_offset = 20

        for i, word in enumerate(suggested_words):
            cv2.putText(frame, f"Suggested word {i + 1}: {word}", (10, y_offset + i * 20),
                        font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    def display_prediction(self, frame, i, label, sign_image):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f'Predicted sign {i}: {label}', (10, 50 + i * 30),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        small_image = cv2.resize(sign_image.reshape(28, 28).astype('float32') / 255, (100, 100))
        small_image = (small_image * 255).astype('uint8')
        frame[10 + i * 110:110 + i * 110, -110:-10] = cv2.cvtColor(small_image, cv2.COLOR_GRAY2RGB)
 
if __name__ == "__main__":
    recognizer = SignRecognizer()
    recognizer.model_check()
    recognizer.video_processor()
