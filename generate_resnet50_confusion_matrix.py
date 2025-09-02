import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Load and preprocess your data (example)
X_train = np.load('X_train.npy')  # Replace with your training data
y_train = np.load('y_train.npy')  # Replace with your training labels
X_test = np.load('X_test.npy')    # Replace with your test data
y_test = np.load('y_test.npy')    # Replace with your test labels

# Resize data to 224x224 (ResNet50 input size)
X_train = tf.image.resize(X_train, [224, 224])
X_test = tf.image.resize(X_test, [224, 224])
y_train = tf.keras.utils.to_categorical(y_train, 10)  # 10 classes
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Load ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze base layers

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)  # 10 classes

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# Save model
model.save('resnet50_model.h5')
print("ResNet50 model saved as 'resnet50_model.h5'.")