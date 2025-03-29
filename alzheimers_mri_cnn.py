import tensorflow as tf
import matplotlib.pyplot as plt
import os


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

# Define paths
data_dir = "dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Image parameters
img_size = (224, 224)
batch_size = 32

# Data Augmentation & Loading
datagen = ImageDataGenerator(
    rescale=1.0/255,        # Normalize pixel values between 0 and 1
    rotation_range=30,      # Increase rotation range
    width_shift_range=0.3,  # Increase width shift range
    height_shift_range=0.3, # Increase height shift range
    shear_range=0.3,        # Increase shearing transformations
    zoom_range=0.3,         # Increase zoom range
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
    subset='training' # Since datagen has a validation_split, this selects the portion of the data for training.
)

val_generator = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Crucially, this is set to False for the test data. This is because we want to evaluate the model on the original order of the test images to get a consistent and accurate evaluation. Shuffling would randomize the order, making it difficult to compare predictions with the true labels.
)

# Load VGG16 model with pre-trained weights
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze the base model

# Build CNN Model
# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Conv2D(128, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dropout(0.5),
#     Dense(4, activation='softmax')
# ])

model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(0.001)),  # Add L2 regularization
    Dropout(0.5),
    Dense(4, activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
epochs = 50
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate Model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")

# Save Model
model.save("alzheimers_mri_classifier.h5")

# Plot Training History
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
