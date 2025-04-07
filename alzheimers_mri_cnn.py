import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

# Enable mixed precision training
mixed_precision.set_global_policy('mixed_float16')
tf.config.optimizer.set_jit(True)

# Define paths
data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Image parameters
img_size = (256, 256)
batch_size = 32

# Custom preprocessing function for MRI images
def preprocess_mri(image):
    # Convert to grayscale if RGB
    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
    
    # Normalize using Z-score normalization
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-7)
    
    # Repeat grayscale channel 3 times for pre-trained model
    image = tf.image.grayscale_to_rgb(image)
    return image

# Data Augmentation & Loading
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_mri,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_mri)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Model Architecture
base_model = EfficientNetB3(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

# Fine-tuning strategy
for layer in base_model.layers[:-100]:
    layer.trainable = False

# Enhanced model architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(4, activation='softmax')
])

# Define the learning rate
initial_learning_rate = 1e-4

# Compile the model with the learning rate
model.compile(
    optimizer=Adam(learning_rate=initial_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=12,
        restore_best_weights=True,
        min_delta=0.001
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1,
        min_lr=1e-6
    )
]

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=100,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Save model
model.save("alzheimers_mri_classifier_improved.keras")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save training history
history_path = "training_history.txt"
with open(history_path, 'w') as f:
    for key in history.history.keys():
        f.write(f"{key}: {history.history[key]}\n")
# Save training history plot
history_plot_path = "training_history_plot.png"
plt.savefig(history_plot_path)