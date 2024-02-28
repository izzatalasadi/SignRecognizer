import time
import os
import logging
import concurrent.futures
import cv2
import mediapipe as mp
import numpy as np
import torch
from keras.models import load_model
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from contextlib import contextmanager

from ModelCreation import ModelCreation

class SignRecognizer:
    def __init__(self):
        # Load GPT-2 model and tokenizer
        self.gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Load sign recognizer model
        self.model_check()

        # Load sign dataset
        model_creation = ModelCreation(auto_run=False)
        self.sign_train = model_creation.sign_train
        self.sign_test = model_creation.sign_test

        # Initialize variables for hand tracking
        self.stable_position_timer = None
        self.stable_position_duration = 0.5
        self.words = []

    def model_check(self, model_filename="sign_recognizer_model.keras"):
        # Check if the model file exists and load it, otherwise create and train the model
        model_path = os.path.join(os.path.curdir, 'model', model_filename)
        logging.info(f"Model path: {model_path}")

        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                logging.info(f"Model '{model_filename}' loaded successfully.")
            else:
                logging.warning(f"Model file {model_filename} not found. Training model in progress ...")
                ModelCreation()
                logging.info(f"Model '{model_filename}' created.")

                self.model = load_model(model_path)
                logging.info(f"Model '{model_filename}' loaded successfully.")

        except FileNotFoundError:
            logging.error(f"Model file '{model_filename}' not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {e}")

    @contextmanager
    def video_capture_context_manager(self, index=0):
        # Context manager for video capture
        cap = cv2.VideoCapture(index)
        try:
            if not cap.isOpened():
                raise ValueError("Unable to open video source")
            yield cap
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def check_stable_fingertip_position(self, fingertip_position):
        # Check if fingertip position is stable for a certain duration inside a 2x2 imaginary box
        

        x, y = fingertip_position
        
        # Check if the difference in fingertip position is within the 2x2 box
        for box in word_boxes:
            x_min, y_min, x_max, y_max = box
            if x_min <= fingertip_position[0] <= x_max and y_min <= fingertip_position[1] <= y_max:
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
        # Get fingertip position using MediaPipe Hands
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

    def pick_letter(self, frame, suggested_labels):
        # Pick a letter from suggested labels based on fingertip position
        unique_suggested_labels = list(set(str(label) for label in suggested_labels))

        try:
            with self.video_capture_context_manager(0) as cap:
                while True:
                    _, frame = cap.read()
                    fingertip_position = self.get_fingertip_position(frame)
                    if fingertip_position:
                        x, y = fingertip_position
                        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)
                        for i, label in enumerate(unique_suggested_labels):
                            cv2.putText(frame, label, (i * 50 , 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                            
                            #fix position
                            if 20 + i * 20 <= y <= 20 + (i + 1) * 20:
                                if self.check_stable_fingertip_position(fingertip_position):
                                    logging.info("Selected label: %s", label)
                                    if label:
                                        self.words.append(label)
                                        break

                    cv2.imshow('Hand Tracking', frame)

                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('c'):
                        break

        except Exception as e:
            logging.error(f"An error occurred in another task: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def video_processor(self):
        try:
            with self.video_capture_context_manager(0) as cap:
                label_to_letter = {i: chr(65 + i) for i in range(26)}
                suggested_labels = []
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
                            with concurrent.futures.ThreadPoolExecutor(max_workers=100000) as executor:
                                future = executor.submit(self.process_hand_rois, hand_rois, frame, label_to_letter)
                                result = future.result()
                                if result not in suggested_labels:
                                    suggested_labels.append(result)

                        logging.info("Suggested Sentence: %s", suggested_labels)

                    cv2.imshow('Sign Language Translation', frame)

                    key = cv2.waitKey(1)

                    if key & 0xFF == ord('q'):
                        break
                    if key & 0xFF == ord('c'):
                        logging.info("Processing paused...")
                        self.pause_processing = True  # Set pause_processing to True
                        word = self.pick_letter(frame, suggested_labels)
                        words.append(word)
                        logging.info(f"word '{word}' added to {words} ")

                        suggested_labels.clear()
                        logging.info(f"Suggested labels list deleted")

                        logging.info("Processing resumed...")
                        self.pause_processing = False  # Set pause_processing to False after resuming

        except IOError as e:
            logging.error(f"IOError occurred: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
        finally:
            cv2.destroyAllWindows()

    def detect_hands(self, frame):
        hands = mp.solutions.hands.Hands(static_image_mode=False,
                                        max_num_hands=2,
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
        suggested_labels = set()
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
                suggested_labels.add(label)
        
        return list(suggested_labels)  

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
