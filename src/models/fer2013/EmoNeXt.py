import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, Add, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Directories
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(current_dir, '../../../data/raw/fer2013'))
model_dir = os.path.abspath(os.path.join(current_dir, '../../../models'))

# Parameters
img_size = (48, 48)
batch_size = 32
epochs = 200

# Data Generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(data_dir, 'train'),
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = test_datagen.flow_from_directory(
    os.path.join(data_dir, 'test'),
    target_size=img_size,
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Convert grayscale images to RGB by repeating the single channel three times
def preprocess_input_rgb(images, labels):
    images = tf.image.grayscale_to_rgb(images)
    return images, labels

# Create TensorFlow datasets from the generators
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, train_generator.num_classes), dtype=tf.float32)
    )
).map(preprocess_input_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: validation_generator,
    output_signature=(
        tf.TensorSpec(shape=(None, img_size[0], img_size[1], 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, validation_generator.num_classes), dtype=tf.float32)
    )
).map(preprocess_input_rgb, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Prefetch for performance optimization
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# ConvNeXt Block
def convnext_block(x, filters, drop_path_rate=0., layer_scale_init_value=1e-6):
    shortcut = x
    # Depthwise Convolution
    x = Conv2D(filters, kernel_size=7, padding='same', groups=filters)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(4 * filters)(x)
    x = Activation('gelu')(x)
    x = Dense(filters)(x)
    
    # Stochastic Depth
    if drop_path_rate > 0.:
        x = tf.keras.layers.Dropout(drop_path_rate)(x, training=True)
    
    # Layer Scale
    gamma = tf.Variable(layer_scale_init_value * tf.ones((filters)), trainable=True)
    x = gamma * x
    
    x = Add()([shortcut, x])
    return x

# EmoNeXt Network
def emonext_net(input_shape, num_classes, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.):
    inputs = Input(shape=input_shape)
    # Stem
    x = Conv2D(dims[0], kernel_size=4, strides=4, padding='same')(inputs)
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Stages
    for i in range(len(depths)):
        for j in range(depths[i]):
            x = convnext_block(x, dims[i], drop_path_rate)
        if i != len(depths) - 1:
            x = LayerNormalization(epsilon=1e-6)(x)
            x = Conv2D(dims[i + 1], kernel_size=2, strides=2, padding='same')(x)
    
    x = GlobalAveragePooling2D()(x)
    
    # Classification head
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model

# Model Initialization
input_shape = (img_size[0], img_size[1], 3)
num_classes = train_generator.num_classes
model = emonext_net(input_shape, num_classes)

# Compile the Model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the Model
model.fit(
    train_dataset,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_dataset,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[early_stopping, reduce_lr]
)

# Save the Model
model.save(os.path.join(model_dir, 'fer2013_emonext_model.keras'))
