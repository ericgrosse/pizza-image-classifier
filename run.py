import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset of pizza and non-pizza images and preprocess it
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_dataset = train_datagen.flow_from_directory(
    directory='images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

# Define the CNN architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10)

# Test the model on a set of test images and evaluate its performance
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_dataset = test_datagen.flow_from_directory(
    directory='images',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy:', accuracy)

# Save the model to a file
model.save('pizza_image_classifier.h5')
