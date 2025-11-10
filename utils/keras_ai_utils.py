# library imports
import random
import os
import tensorflow as tf
tf.config.run_functions_eagerly(True)       # enable eager execution due to issues with some tf.data operations in graph mode
import matplotlib.pyplot as plt
import numpy as np
from tqdm.keras import TqdmCallback
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, BatchNormalization, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

# local imports
from utils.main_utils import (
    display_metrics, 
    display_plot_roc, 
    get_file_names, 
    handle_error
)


def build_keras_model(
        train_generator,
        validation_generator,
        optimizer,
        kernel_initializer,
        filter_base=32, 
        unit_base=128, 
        output_units=1, 
        conv_block_num=4, 
        dense_block_num=2, 
        hidden_activation='relu', 
        padding='same', 
        strides=(1, 1), 
        kernel_size=(3, 3),
        kernel_regularizer=l2(0.001),
        img_w=64, 
        img_h=64, 
        n_channels=3,
        loss='binary_crossentropy',
        metrics=['accuracy'],
        model_name='keras_model',
        n_epochs=3,
        steps_per_epoch=None,
        validation_steps=None,
        batch_size=128,
        lr=0.001,
        dropout=.04,
        device='cpu',
        tqdm_verbose=1,
        verbose=0,
        output_activation='sigmoid',
        pool_size=2,
        pool_strides=2
    ):
    # batch normalization for stability, dropout for regularization, and a sigmoid output for binary classification.
    try:
        print('Building Keras model...')

        # initialize the Sequential model
        model = Sequential()

        # add input layer
        model.add(tf.keras.Input(shape=(                    # input layer: specify input shape
            img_w,                                          # image width (e.g., 64)
            img_h,                                          # image height (e.g., 64)
            n_channels                                      # number of channels (e.g., 3 for RGB)
            )))

        # add convolutional blocks
        for conv_block in range(conv_block_num):
            filters = filter_base * (2 ** conv_block)
            model.add(                                      # convolutional block 1: initial feature extraction
                Conv2D(
                    filters=filters,                        # number of filters for initial feature extraction (e.g., 32)
                    kernel_size=kernel_size,                # kernel size for convolution (e.g., (5, 5))
                    padding=padding,                        # padding type (e.g., 'same')
                    strides=strides,                        # stride for convolution (e.g., (1, 1))
                    kernel_initializer=kernel_initializer,  # kernel initializer (e.g., HeUniform)
                    kernel_regularizer=kernel_regularizer,  # L2 regularization to prevent overfitting (e.g., 0.001)
                )                                           
            )
            model.add(BatchNormalization())                     # BatchNormalization: normalize outputs to stabilize training
            model.add(
                tf.keras.layers.Activation(hidden_activation)   # Activation: apply activation function after batch normalization
            )
            model.add(                                          # MaxPooling2D: downsample spatial dimensions
                MaxPooling2D(
                    pool_size=pool_size,                        # size of the pooling window (e.g., 2 for 2x2)
                    strides=pool_strides                        # step size of the pooling window (e.g., 2)
                )
            )

        # transition to fully connected layers
        model.add(GlobalAveragePooling2D())                     # GlobalAveragePooling2D: average pool each feature map to a single value

        # add dense blocks
        for dense_block in range(dense_block_num):
            model.add(                                          # fully connected blocks 1-n: dense layers with increasing neurons and regularization
                Dense(
                    units=unit_base // (2 ** dense_block),      # number of neurons in dense layer (decreased to unit_base / (2 ** dense_block))
                    kernel_initializer=kernel_initializer,      # kernel initializer (e.g., HeUniform)
                    kernel_regularizer=kernel_regularizer       # L2 regularization to prevent overfitting (e.g., 0.001)
                )
            )
            
            model.add(BatchNormalization())                     # BatchNormalization: normalize outputs
            model.add(
                tf.keras.layers.Activation(hidden_activation)   # Activation: apply activation function after batch normalization
            )
            model.add(Dropout(dropout))                         # Dropout: drop specified rate of neurons to prevent overfitting

        # add output layer
        model.add(                                              # output layer
            Dense(
                units=output_units,                             # number of neurons for output layer (e.g., 1 for binary classification)
                activation=output_activation                    # Sigmoid activation for binary classification
            )                                                   
        )

        # compile the model
        model.compile(                                          # compile the model with specified optimizer, loss function, and metrics
            optimizer=optimizer,                                # optimizer for training (e.g., Adam)
            loss=loss,                                          # loss function for training (e.g., binary_crossentropy)
            metrics=metrics                                     # metrics to monitor during training (e.g., accuracy)
        )

        # set up model checkpointing
        checkpoint_cb = ModelCheckpoint(
            filepath=model_name,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=tqdm_verbose
        )

        # train the model
        print(f"Training on : ==={device}=== with batch size: {batch_size} & lr: {lr}")
        fit = model.fit(
            train_generator, 
            epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint_cb, TqdmCallback(verbose=tqdm_verbose)], 
            verbose=verbose
        )

        # return the model and training history
        return model, fit
    except Exception as e:
        handle_error(e, def_name='build_keras_model')


def configure_keras_for_performance(train_ds, val_ds):
    try:
        print('Configuring Keras datasets for performance...')
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
        ])

        # apply augmentation to the training dataset using map
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

        # configure for performance
        AUTOTUNE = tf.data.AUTOTUNE
        configured_train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        configured_val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        # return the configured datasets
        return configured_train_ds, configured_val_ds
    except Exception as e:
        handle_error(e, def_name='configure_keras_for_performance')


def create_generators(dataset_path, target_size, batch_size, class_mode, rescale, rotation_range, width_shift_range, height_shift_range, shear_range, zoom_range, horizontal_flip, fill_mode, validation_split):
    datagen = ImageDataGenerator(
        rescale=rescale,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        horizontal_flip=horizontal_flip,
        fill_mode=fill_mode,
        validation_split=validation_split
    )
    train_generator = datagen.flow_from_directory(
        directory=dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        directory=dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=class_mode,
        subset='validation'
    )
    return train_generator, validation_generator


def create_keras_datasets(base_dir, img_size, batch_size):
    try:
        print('Creating Keras datasets...')

        # create a training dataset
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            labels='inferred',          # labels are generated from directory structure
            label_mode='int',           # labels are encoded as integers (0, 1, ...)
            validation_split=0.2,       # reserve 20% of images for validation
            subset='training',          # this is the training set
            seed=1337,                  # shuffle seed for reproducible splits
            image_size=img_size,
            batch_size=batch_size
        )
        # create a validation dataset
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_dir,
            labels='inferred',
            label_mode='int',
            validation_split=0.2,
            subset='validation',
            seed=1337,
            image_size=img_size,
            batch_size=batch_size
        )
        return train_ds, val_ds
    except Exception as e:
        handle_error(e, def_name='create_keras_datasets')


def display_custom_keras_batch(data_generator, batch_size, title=None):
    try:
        print('Displaying a custom data batch...')

        # get one batch data
        images, labels = next(data_generator)
        print(f'Images batch shape: {images.shape}')
        print(f'Labels batch shape: {labels.shape}')
        
        # display the images in the batch
        plt.figure(figsize=(12, 6))
        for i in range(batch_size):
            plt.subplot(2, 4, i + 1)
            plt.imshow(images[i])
            plt.title(f'{title if title else ""}  Label: {int(labels[i])}')
            plt.axis('off')
    except Exception as e:
        handle_error(e, def_name='display_custom_keras_batch')


def display_keras_batch(batch_size, dataset, title=None):
    try:
        print('Displaying a Keras data batch...')
        plt.figure(figsize=(10, 10))
        for images, labels in dataset.take(1):  # take one batch
            for i in range(batch_size):
                plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype('uint8'))  # images are loaded as float32, so we convert to uint8 for display
                plt.title(f'{title if title else ""}  Label: {int(labels[i])}')
                plt.axis('off')
    except Exception as e:
        handle_error(e, def_name='display_keras_batch')


def display_keras_history(validation_generator, model, fit):
    try:
        print('Displaying Keras model accuracy and loss history...')

        # calculate accuracy on the validation set
        steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
        all_preds = []
        all_labels = []
        for _ in range(steps):
            # get one batch data
            images, labels = next(validation_generator)
            preds = model.predict(images)
            preds = (preds > 0.5).astype(int).flatten() 
            all_preds.extend(preds)
            all_labels.extend(labels)
        accuracy = accuracy_score(all_labels, all_preds)
        print(f"Accuracy Score: {accuracy:.4f}")

        # create a figure with a subplot - FIX: unpack the tuple correctly
        _, axs = plt.subplots(figsize=(8, 6))

        # plot accuracy on the first subplot, then display
        axs.plot(fit.history['accuracy'], label='Training Accuracy')
        axs.plot(fit.history['val_accuracy'], label='Validation Accuracy')
        axs.set_title('Model Accuracy')
        axs.set_xlabel('Epochs')
        axs.set_ylabel('Accuracy')
        axs.legend()
        axs.grid(True)
        plt.tight_layout()
        plt.show()

        # create a new figure for loss
        _, axs = plt.subplots( figsize=(8, 6))

        # plot loss on the second subplot
        axs.plot(fit.history['loss'], label='Training Loss')
        axs.plot(fit.history['val_loss'], label='Validation Loss')
        axs.set_title('Model Loss')
        axs.set_xlabel('Epochs')
        axs.set_ylabel('Loss')
        axs.legend()
        axs.grid(True)

        plt.tight_layout()
        plt.show()
    except Exception as e:
        handle_error(e, def_name='display_model_accuracy_and_loss_history')


def evaluate_keras_model(model, validation_generator, device, model_name):
    try:
        print('Evaluating Keras model...')

        # get true labels and predicted probabilities
        y_true = []
        y_prob = []
        steps = int(np.ceil(validation_generator.samples / validation_generator.batch_size))
        for _ in range(steps):
            images, labels = next(validation_generator)
            preds = model.predict(images)
            y_true.extend(labels)
            y_prob.extend(preds.flatten())
        
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        y_pred = (y_prob > 0.5).astype(int)

        # get class labels
        class_labels = list(validation_generator.class_indices.keys())

        # display metrics and ROC curve
        display_metrics(y_true, y_pred, y_prob, class_labels, model_name)
        display_plot_roc(y_true, y_prob, model_name)
    except Exception as e:
        handle_error(e, def_name='evaluate_keras_model')


def keras_custom_data_generator(image_paths, labels, batch_size, target_size=(64, 64)):
    try:
        print('Generating custom data batches...')
        num_samples = len(image_paths)
        while True: 
            # shuffle data at the beginning of each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            shuffled_paths = np.array(image_paths)[indices]
            shuffled_labels = np.array(labels)[indices]
            
            # generate batch data
            for offset in range(0, num_samples, batch_size):
                batch_paths = shuffled_paths[offset:offset+batch_size]
                batch_labels = shuffled_labels[offset:offset+batch_size]
                
                # load and preprocess images from the batch
                batch_images = []
                for path in batch_paths:
                    img = tf.keras.utils.load_img(path, target_size=target_size)
                    img_array = tf.keras.utils.img_to_array(img)
                    batch_images.append(img_array)
                
                # normalize and yield the batch data
                yield np.array(batch_images) / 255.0, np.array(batch_labels)
    except Exception as e:
        handle_error(e, def_name='keras_custom_data_generator')


def set_tf_processing_env(dataset_path):
    try:
        print('Setting TensorFlow processing environment...')

        # set environment variables for TensorFlow
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # disable oneDNN optimizations (0: disable, 1: enable)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress TensorFlow logging (0: all, 1: warnings+, 2: errors+, 3: fatal only)

        # check for GPU availability
        gpu_list = tf.config.list_physical_devices('GPU')
        device = 'gpu' if gpu_list !=[] else 'cpu'
        print(f'Device available for training: {device}')

        # set seed for reproducibility
        seed_value = 1
        set_tf_seed(seed_value)

        # get file names
        fnames = get_file_names(dataset_path)

        # return the device and file names
        return device, fnames
    except Exception as e:
        handle_error(e, def_name='set_tf_processing_env')


def set_tf_seed(seed_value=42):
    try:
        print(f'Setting TensorFlow random seed to {seed_value}...')
        random.seed(seed_value)
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
    except Exception as e:
        handle_error(e, def_name='set_tf_seed')