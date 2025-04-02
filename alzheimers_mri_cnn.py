import tensorflow as tf
import matplotlib.pyplot as plt
import os
from keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Flatten = tf.keras.layers.Flatten
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Adam = tf.keras.optimizers.Adam
VGG16 = tf.keras.applications.VGG16
EarlyStopping = tf.keras.callbacks.EarlyStopping
l2 = tf.keras.regularizers.l2
GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
MobileNetV2 = tf.keras.applications.MobileNetV2

tf.config.optimizer.set_jit(True) # Enable XLA

# Define paths
data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Image parameters
img_size = (224, 224)
batch_size = 64  # Increased batch size

# Data Augmentation & Loading
	# Contrast & Histogram Equalization: Since MRI scans have grayscale intensity variations, use Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance the contrast.
	# Normalize Images with Z-score: Instead of rescale=1.0/255, try (image - mean) / std normalization to handle variations in brightness.
datagen = ImageDataGenerator(
    rescale=1.0/255,        # Normalize pixel values between 0 and 1
    rotation_range=20,      # Decreased rotation range
    width_shift_range=0.2,  # Decreased width shift range
    height_shift_range=0.2, # Decreased height shift range
    shear_range=0.2,        # Decreased shearing transformations
    zoom_range=0.2,         # Decreased zoom range
    horizontal_flip=True,   # Flip images horizontally
    brightness_range=[0.8, 1.2],  # Adjust brightness
    validation_split=0.2    # Split 20% of training data for validation
)

# This is a function that makes it easy to load images from a directory. 
# It automatically organizes the data based on subfolders within the directory.
train_generator = datagen.flow_from_directory( 
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification
    subset='training', # Since datagen has a validation_split, this selects the portion of the data for training.

)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
)


test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,  # Crucially, this is set to False for the test data. This is because we want to evaluate the model on the original order of the test images to get a consistent and accurate evaluation. Shuffling would randomize the order, making it difficult to compare predictions with the true labels.
)

# Load VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
base_model.trainable = True  # Unfreeze the last few layers of the base model

# Build CNN Model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Add L2 regularization
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)  # Increased learning rate

# Early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss', # Monitor validation loss
    patience=5, # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True # Keep the best model weights
)  


# Train Model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping], 
    verbose=1, # Show progress bar
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Save Model
model.save("alzheimers_mri_classifier.keras")

# Plot Training History
plt.figure(figsize=(12,5))
plt.subplot(1,2,1) # 1 row, 2 columns, 1st subplot
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2) # 1 row, 2 columns, 2nd subplot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss') # Used to detect overfitting (if validation loss starts increasing while training loss keeps decreasing) 
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
