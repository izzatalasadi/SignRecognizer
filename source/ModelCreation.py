
import os
import logging
from keras.models import Sequential
from keras.layers import (BatchNormalization, Conv2D, Dense,
                          Dropout, Flatten, MaxPooling2D)
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from DataPreparation import DataPreparation


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
        path = os.path.join(os.path.curdir, 'model')
        self.model.save(os.path.join(path, self.model_filename))
        logging.info(f"Model saved in: {os.path.join(path, self.model_filename)}")
