import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random
import numpy as np

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values (scale between 0 and 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Split dataset into 80% training and 20% validation
split = int(0.8 * len(x_train))
x_train, x_val = x_train[:split], x_train[split:]
y_train, y_val = y_train[:split], y_train[split:]

# Define ImageDataGenerator for Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,       # Rotate images up to 15 degrees
    width_shift_range=0.3,   # Shift images horizontally by 10%
    height_shift_range=0.3,  # Shift images vertically by 10%
    horizontal_flip=True     # Randomly flip images horizontally
)

# Fit the generator to training data
datagen.fit(x_train)

# Define the CNN Model
def create_cnn():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    return model

# Create and compile the model
model = create_cnn()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with augmented data
epochs = 20
batch_size = 32

history_aug = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),  # Augmented Data
    validation_data=(x_val, y_val),
    epochs=epochs
)

#early stopping incase of overfitting
#early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#history_aug = model.fit(
#    datagen.flow(x_train, y_train, batch_size=batch_size),
 #   validation_data=(x_val, y_val),
  #  epochs=50,  # Higher value, but will stop early if needed
   # callbacks=[early_stopping]
#)

# Plot training & validation accuracy
plt.plot(history_aug.history['accuracy'], label='Train Accuracy')
plt.plot(history_aug.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy (With Augmentation)')
plt.show()

# Plot training & validation loss
plt.plot(history_aug.history['loss'], label='Train Loss')
plt.plot(history_aug.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss (With Augmentation)')
plt.show()

# Evaluate model on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy with Augmentation: {test_acc:.4f}")
print(f"Test Loss with Augmentation: {test_loss:.4f}")

# Get predictions
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Compute confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Select random test images
num_images = 5
indices = random.sample(range(len(x_test)), num_images)
sample_images = x_test[indices]
sample_labels = y_test[indices]
sample_preds = model.predict(sample_images)
sample_preds_classes = np.argmax(sample_preds, axis=1)

# Plot the images with predicted labels
fig, axes = plt.subplots(1, num_images, figsize=(15,3))
for i, ax in enumerate(axes):
    ax.imshow(sample_images[i])
    ax.set_title(f"Pred: {sample_preds_classes[i]}")
    ax.axis("off")
plt.show()
