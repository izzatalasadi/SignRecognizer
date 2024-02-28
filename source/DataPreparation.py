import pandas as pd
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


class DataPreparation():
    def __init__(self, train_data="sign_mnist_train.csv", test_data="sign_mnist_test.csv") -> None:
        self.sign_train = pd.read_csv(os.path.join(os.path.curdir, 'data',train_data))
        self.sign_test = pd.read_csv(os.path.join(os.path.curdir, 'data',test_data))
        self.num_classes = len(self.sign_train['label'].unique()) + 1
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
        
    