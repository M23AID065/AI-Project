import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy

# Enhanced Data Augmentation
train_datagen = ImageDataGenerator(
    zoom_range=0.3,
    shear_range=0.3,
    rotation_range=30,
    brightness_range=[0.2, 1.0],
    horizontal_flip=True,
    rescale=1./255
)

# Train and validation data generators
train_data = train_datagen.flow_from_directory(
    directory=r"C:\\Users\\lohit ramaraju\\OneDrive\\Desktop\\IITJ\\AI\\Project\\DataSet\\Dataset\\train",
    target_size=(224, 224),
    batch_size=32,
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    directory=r"C:\\Users\\lohit ramaraju\\OneDrive\\Desktop\\IITJ\\AI\\Project\\DataSet\\Dataset\\test",
    target_size=(224, 224),
    batch_size=32
)

# Visualize images in training data
t_img, label = next(train_data)

def plotImages(img_arr, label):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(img_arr[i])
        plt.title(f"Label: {np.argmax(label[i])}")
        plt.axis('off')
    plt.show()

plotImages(t_img, label)

# Model Setup: Using MobileNet with fine-tuning
base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freezing all layers except the last 20 for fine-tuning
for layer in base_model.layers[:-20]:  
    layer.trainable = False

# Adding custom layers on top of MobileNet
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to reduce overfitting
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(7, activation='softmax')(x)  # Adjust output units to the number of classes

# Compile model
model = Model(base_model.input, output)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Define Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Train the Model
history = model.fit(
    train_data,
    steps_per_epoch=10,
    epochs=100,
    validation_data=val_data,
    validation_steps=10
    callbacks=[early_stopping, reduce_lr]
)

# Plot training and validation accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

plot_training_history(history)


# Load and predict on a new image
path = r"C:\Users\lohit ramaraju\OneDrive\Desktop\IITJ\AI\Project\emotion-based-music-player\test.jpg"
img = load_img(path, target_size=(224, 224))
i = img_to_array(img) / 255.0  # Normalize the image
input_arr = np.array([i])
input_arr.shape

# Prediction
pred = np.argmax(model.predict(input_arr))
print(f"The image is of {op[pred]}")

# Display the input image
plt.imshow(input_arr[0])
plt.title("Input Image")
plt.show()
