import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../../../data/raw/ck+'))
model_dir = os.path.abspath(os.path.join(current_dir, '../../../models'))

# Parameters
img_size = (48, 48)
batch_size = 32
epochs = 50
validation_split = 0.2 

# Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split  # Use for splitting the dataset into train and validation
)

# Flow from directory with tf.data.Dataset
def data_generator(datagen, directory, img_size, batch_size, subset):
    generator = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        color_mode='grayscale',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset=subset
    )
    
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_size[0], img_size[1], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, generator.num_classes), dtype=tf.float32)
        )
    )
    
    dataset = dataset.repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = data_generator(datagen, data_dir, img_size, batch_size, subset='training')
validation_dataset = data_generator(datagen, data_dir, img_size, batch_size, subset='validation')

# Print number of samples
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation'
)

print(f"Number of training samples: {train_generator.samples}")
print(f"Number of validation samples: {validation_generator.samples}")

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // batch_size
validation_steps = validation_generator.samples // batch_size

# Define simple CNN model
simple_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
simple_cnn.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model
history = simple_cnn.fit(
    train_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_dataset,
    validation_steps=validation_steps,
    epochs=epochs,
    # callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model on the validation set
val_loss, val_accuracy = simple_cnn.evaluate(validation_dataset, steps=validation_steps)
print(f"Validation loss: {val_loss}")
print(f"Validation accuracy: {val_accuracy}")

# Save the model
simple_cnn.save(os.path.join(model_dir, 'ck+_simple_cnn_model.keras'))
