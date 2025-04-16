import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from PIL import UnidentifiedImageError, Image
import matplotlib.pyplot as plt

# Enable mixed precision training
def configure_environment():
    """
    Configures the environment for mixed precision training and enables JIT compilation.
    """
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)

# Define paths
DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

def clean_directory(folder):
    """
    Cleans the specified directory by removing unreadable image files.

    Args:
        folder (str): The path to the directory to clean.
    """
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                Image.open(path).verify()
            except (UnidentifiedImageError, OSError):
                print(f"Removing unreadable image: {path}")
                os.remove(path)

# Image parameters
IMG_SIZE = (256, 256)
BATCH_SIZE = 32

def preprocess_mri(image):
    """
    Preprocesses MRI images by converting to grayscale, normalizing, and converting back to RGB.

    Args:
        image (tf.Tensor): The input image tensor.

    Returns:
        tf.Tensor: The preprocessed image tensor.
    """
    # Check if the image has any dimensions before proceeding
    if tf.size(image) == 0:
        # Handle the case of an empty image, e.g., return a black image
        return tf.zeros_like(image, dtype=image.dtype)

    if image.shape[-1] == 3:
        image = tf.image.rgb_to_grayscale(image)
    mean = tf.reduce_mean(image)
    std = tf.math.reduce_std(image)
    image = (image - mean) / (std + 1e-7)
    image = tf.image.grayscale_to_rgb(image)
    return image

def safe_load_img(path, target_size=None, color_mode='rgb'):
    """
    Safely loads an image from the specified path, handling unreadable images.

    Args:
        path (str): The path to the image file.
        target_size (tuple): The target size for the image.
        color_mode (str): The color mode for the image.

    Returns:
        PIL.Image or None: The loaded image or None if unreadable.
    """
    try:
        return tf.keras.utils.load_img(path, target_size=target_size, color_mode=color_mode)
    except UnidentifiedImageError:
        print(f"Skipping unreadable image: {path}")
        return None

def create_data_generators():
    """
    Creates data generators for training, validation, and testing with data augmentation.

    Returns:
        tuple: A tuple containing the training, validation, and test data generators.
    """
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_mri,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,
        validation_split=0.2,
    )

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_mri)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def build_model():
    """
    Builds and returns a Sequential model with EfficientNetB3 as the base.

    Returns:
        tf.keras.Model: The constructed model.
    """
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(256, 256, 3)
    )

    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        BatchNormalization(),
        Dense(1536, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(768, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(384, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(4, activation='softmax')
    ])

    return model

def compile_model(model):
    """
    Compiles the model with AdamW optimizer and categorical crossentropy loss.

    Args:
        model (tf.keras.Model): The model to compile.
    """
    initial_learning_rate = 1e-4
    model.compile(
        optimizer=AdamW(learning_rate=initial_learning_rate, weight_decay=1e-5),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

def get_callbacks():
    """
    Returns a list of callbacks for model training.

    Returns:
        list: A list of Keras callbacks.
    """
    return [
        ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
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

def train_model(model, train_generator, val_generator):
    """
    Trains the model using the provided data generators.

    Args:
        model (tf.keras.Model): The model to train.
        train_generator (DirectoryIterator): The training data generator.
        val_generator (DirectoryIterator): The validation data generator.

    Returns:
        History: The training history object.
    """
    callbacks = get_callbacks()
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, test_generator):
    """
    Evaluates the model on the test data generator.

    Args:
        model (tf.keras.Model): The model to evaluate.
        test_generator (DirectoryIterator): The test data generator.
    """
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc:.4f}")

def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss over epochs.

    Args:
        history (History): The training history object.
    """
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

def save_training_history(history):
    """
    Saves the training history to a text file and a plot image.

    Args:
        history (History): The training history object.
    """
    history_path = "training_history.txt"
    with open(history_path, 'w', encoding="utf-8") as f:
        for key in history.history.keys():
            f.write(f"{key}: {history.history[key]}\n")

    history_plot_path = "training_history_plot.png"
    plt.savefig(history_plot_path)
    plt.close()

def get_img_array(img_path, size):
    """
    Loads an image and preprocesses it into an array suitable for model input.

    Args:
        img_path (str): The path to the image file.
        size (tuple): The target size for the image.

    Returns:
        np.ndarray or None: The preprocessed image array or None if unreadable.
    """
    img = safe_load_img(img_path, target_size=size)
    if img is None:
        return None
    array = tf.keras.utils.img_to_array(img)
    array = preprocess_mri(array)
    return np.expand_dims(array, axis=0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates a Grad-CAM heatmap for the given image array and model.

    Args:
        img_array (np.ndarray): The preprocessed image array.
        model (tf.keras.Model): The model to use for Grad-CAM.
        last_conv_layer_name (str): The name of the last convolutional layer.
        pred_index (int, optional): The index of the predicted class.

    Returns:
        np.ndarray: The generated heatmap.
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    Saves and displays a Grad-CAM heatmap superimposed on the original image.

    Args:
        img_path (str): The path to the original image.
        heatmap (np.ndarray): The Grad-CAM heatmap.
        cam_path (str): The path to save the superimposed image.
        alpha (float): The transparency factor for the heatmap overlay.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)

def main():
    """
    Main function to execute the training, evaluation, and Grad-CAM visualization.
    """
    configure_environment()
    clean_directory(TRAIN_DIR)
    clean_directory(TEST_DIR)

    train_generator, val_generator, test_generator = create_data_generators()
    model = build_model()
    compile_model(model)

    history = train_model(model, train_generator, val_generator)
    evaluate_model(model, test_generator)

    model.save("alzheimers_mri_classifier_improved.keras")
    plot_training_history(history)
    save_training_history(history)

    # Example usage of Grad-CAM
    sample_img_path = val_generator.filepaths[0]
    img_array = get_img_array(sample_img_path, size=IMG_SIZE)
    if img_array is not None:
        last_conv_layer_name = [layer.name for layer in model.layers[0].layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
        save_and_display_gradcam(sample_img_path, heatmap, cam_path="gradcam_result.jpg")
        print("Grad-CAM saved to gradcam_result.jpg")

if __name__ == "__main__":
    main()