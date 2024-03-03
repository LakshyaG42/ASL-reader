import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import tensorflow
import torch 

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

# Define the path to the dataset
data_path = 'kaggle\input\\asl_dataset\\asl_dataset'

# Create a dictionary of relationship between label and sign
categories = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "a", 11: "b", 12: "c", 13: "d", 14: "e", 15: "f", 16: "g", 17: "h", 18: "i",
    19: "j", 20: "k", 21: "l", 22: "m", 23: "n", 24: "o", 25: "p", 26: "q", 27: "r",
    28: "s", 29: "t", 30: "u", 31: "v", 32: "w", 33: "x", 34: "y", 35: "z"
}

# Initialize lists to store file paths, images, and labels
file_list = []
image_list = []
label_list = []

# Loop through directories in the dataset
for directory in os.listdir(data_path):
    directory_path = os.path.join(data_path, directory)
    if os.path.isdir(directory_path) and directory != 'asl_dataset':
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            file_list.append(file_path)
            label_list.append(directory)
            img = plt.imread(file_path)
            image_list.append(np.array(img))

# Create a DataFrame from the collected data
df = pd.DataFrame({'file': file_list, 'image': image_list, 'label': label_list})

# Split dataframe into train, test, and validation sets
x_train, x_test0, y_train, y_test0 = train_test_split(df['file'], df['label'], test_size=0.25, random_state=42)
x_test, x_val, y_test, y_val = train_test_split(x_test0, y_test0, test_size=0.5, random_state=42)
train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)
val = pd.concat([x_val, y_val], axis=1).reset_index(drop=True)

# Print shapes of train, test, and validation sets
print(np.shape(train))
print(np.shape(test))
print(np.shape(val))

# Normalize image data and transform into train, test, and validation datasets
image_size = 128
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0 / 255)

train_data = datagen.flow_from_dataframe(dataframe=train, x_col='file', y_col='label', target_size=(image_size, image_size),
                                         batch_size=batch_size, class_mode='categorical')

test_data = datagen.flow_from_dataframe(dataframe=test, x_col='file', y_col='label', target_size=(image_size, image_size),
                                        shuffle=False, batch_size=batch_size, class_mode='categorical')

val_data = datagen.flow_from_dataframe(dataframe=val, x_col='file', y_col='label', target_size=(image_size, image_size),
                                       shuffle=False, batch_size=batch_size, class_mode='categorical')

# Initialize model
model = Sequential()

model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(image_size, image_size, 3)))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(padding='same'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(36, activation='softmax'))

model.summary()

# Initialize callbacks
earlystop = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5, restore_best_weights=True, verbose=0)
reducelr = ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.5, verbose=1)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN model on training data
classifier = model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[earlystop, reducelr], verbose=1)

# Evaluate model
train_loss, train_accuracy = model.evaluate(train_data)
print('Train Accuracy =', train_accuracy)
print('Train Loss =', train_loss)

val_loss, val_accuracy = model.evaluate(val_data)
print('Validation Accuracy =', val_accuracy)
print('Validation Loss =', val_loss)


# Save the model
model.save('asl_model.h5')
