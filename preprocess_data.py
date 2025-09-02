import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Define data directory (adjust to your PlantVillage dataset path)
data_dir = 'D:/LeafScanner/PlantVillage'  # Replace with your dataset path
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Data augmentation and preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.15)

# Train generator
train_generator = datagen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='training')
X_train, y_train = next(train_generator)  # Get one batch
for _ in range(train_generator.samples // 32 - 1):  # Append remaining batches
    X_batch, y_batch = next(train_generator)
    X_train = np.concatenate((X_train, X_batch))
    y_train = np.concatenate((y_train, y_batch))
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)

# Test generator
test_generator = datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', subset='validation')
X_test, y_test = next(test_generator)  # Get one batch
for _ in range(test_generator.samples // 32 - 1):  # Append remaining batches
    X_batch, y_batch = next(test_generator)
    X_test = np.concatenate((X_test, X_batch))
    y_test = np.concatenate((y_test, y_batch))
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

print("Data saved as X_train.npy, y_train.npy, X_test.npy, y_test.npy.")