import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, \
    Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import path as pathes
import cv2
import os
import glob
import params

import numpy as np


def dice_coefficient(y_true, y_pred):
    """
    Calculate the Dice coefficient, a metric used for evaluating segmentation performance.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice coefficient value.
    """
    smooth = 1e-15

    y_true_f = tf.keras.backend.flatten( y_true )
    y_pred_f = tf.keras.backend.flatten( y_pred )

    y_true_f = tf.cast( y_true_f, tf.float32 )
    intersection = tf.keras.backend.sum( y_true_f * y_pred_f )
    return (2. * intersection + smooth) / (tf.keras.backend.sum( y_true_f ) + tf.keras.backend.sum( y_pred_f ) + smooth)


def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss, which is 1 minus the Dice coefficient.

    Args:
        y_true (tensorflow.Tensor): True binary labels.
        y_pred (tensorflow.Tensor): Predicted binary labels.

    Returns:
        float: Dice loss value.
    """
    return 1. - dice_coefficient( y_true, y_pred )

def conv_block(inputs, num_filters):
    """
    Convolutional block consisting of two convolutional layers with batch normalization and ReLU activation.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for convolutional layers.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2D( num_filters, 3, padding="same" )( inputs )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    x = Conv2D( num_filters, 3, padding="same" )( x )
    x = BatchNormalization()( x )
    x = Activation( "relu" )( x )

    return x


def encoder_block(inputs, num_filters):
    """
    Encoder block comprising a convolutional block followed by max pooling.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        Tuple[tensorflow.Tensor, tensorflow.Tensor]: Output tensors from the convolutional block and max pooling.
    """
    x = conv_block( inputs, num_filters )
    p = MaxPool2D( (2, 2) )( x )
    return x, p


def decoder_block(inputs, skip_features, num_filters):
    """
    Decoder block consisting of transposed convolution, concatenation with skip connections, and convolutional block.

    Args:
        inputs (tensorflow.Tensor): Input tensor.
        skip_features (tensorflow.Tensor): Skip connection tensor from encoder block.
        num_filters (int): Number of filters for the convolutional block.

    Returns:
        tensorflow.Tensor: Output tensor.
    """
    x = Conv2DTranspose( num_filters, 2, strides=2, padding="same" )( inputs )
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x


def build_unet(input_shape):
    """
    Function to build the U-Net model architecture.

    Args:
        input_shape (tuple): Shape of the input tensor (height, width, channels).

    Returns:
        tensorflow.keras.Model: U-Net model.
    """
    inputs = Input( input_shape )

    s1, p1 = encoder_block( inputs, 32 )
    s2, p2 = encoder_block( p1, 64 )
    s3, p3 = encoder_block( p2, 128 )
    s4, p4 = encoder_block( p3, 256 )

    b1 = conv_block( p4, 512 )

    d1 = decoder_block( b1, s4, 256 )
    d2 = decoder_block( d1, s3, 128 )
    d3 = decoder_block( d2, s2, 64 )
    d4 = decoder_block( d3, s1, 32 )

    outputs = Conv2D( 1, 1, padding="same", activation="sigmoid" )( d4 )

    model = Model( inputs, outputs, name="UNET" )
    return model


def positiv_negativ_diagnosis(mask_path):
    """
    Function to determine if there is a positive value in the mask.

    This function reads the mask image from the specified path, finds the maximum pixel value,
    and returns 1 if the found value is greater than zero, and 0 otherwise.

    Parameters:
    mask_path (str): Path to the mask image.

    Returns:
    int: 1 if there is a value greater than zero in the mask (positive value), and 0 otherwise (negative value).
    """

    value = np.max( cv2.imread( mask_path ) )


    if value > 0:
        return 1
    else:
        return 0


def create_mri_mask_df():
    """
    Function to create a DataFrame containing MRI image and mask paths along with their corresponding labels.

    This function scans a directory for subdirectories containing MRI images and their corresponding masks,
    creates a DataFrame with the paths and labels, and splits the data into training and validation sets.

    Returns:
    pd.DataFrame: DataFrame containing patient IDs, image paths, mask paths, diagnosis labels, and set labels.
    """
    print( '----Data preparation-----------------------------' )
    data_map = []

    # Iterate over subdirectories in the DATA_PATH directory
    for sub_dir_path in glob.glob( pathes.DATA_PATH + "*" ):
        if os.path.isdir( sub_dir_path ):
            dirname = sub_dir_path.split( "\\" )[-1]
            # Iterate over files in the current subdirectory
            for filename in os.listdir( sub_dir_path ):
                image_path = sub_dir_path + "\\" + filename
                data_map.extend( [dirname, image_path] )

    # Create a DataFrame with directory names and file paths
    df = pd.DataFrame( {"dirname": data_map[::2],
                        "path": data_map[1::2]} )

    # Separate image paths and mask paths
    df_imgs = df[~df['path'].str.contains( "mask" )]
    df_masks = df[df['path'].str.contains( "mask" )]

    # Sort image paths and mask paths
    imgs = sorted( df_imgs["path"].values, key=lambda x: int( x[params.BASE_LEN:-params.END_IMG_LEN] ) )
    masks = sorted( df_masks["path"].values, key=lambda x: int( x[params.BASE_LEN:-params.END_MASK_LEN] ) )

    # Extract patient IDs from image paths
    dirname = []
    for path in imgs:
        patient_id = path.split( '\\' )[3]
        dirname.append( patient_id )

    df = pd.DataFrame( {"patient": dirname,
                        "image_path": imgs,
                        "mask_path": masks} )

    # Apply the positiv_negativ_diagnosis function to determine the diagnosis
    df["diagnosis"] = df["mask_path"].apply( lambda m: positiv_negativ_diagnosis( m ) )
    df['patient_id'] = df['patient'].apply( lambda x: '_'.join( x.split( '_' )[:3] ) )

    # Split the data into training and validation sets based on patient IDs
    train_patients, val_patients = train_test_split( df["patient_id"].unique(), test_size=0.2,
                                                     random_state=params.seed )
    df["set"] = np.where( df["patient_id"].isin( train_patients ), "train", "val" )

    return df


def create_generators(df):
    """
    Function to create data generators for training and validation images and masks.

    This function configures and initializes ImageDataGenerators for data augmentation
    and normalization, and then creates data generators for training and validation sets
    using the provided DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing image and mask paths along with set labels.

    Returns:
    tuple: A tuple containing training data generator and validation data generator.
    """
    print( '----Image and mask augmentation--------------------------' )

    # Configure data generator for training images
    train_image_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=4,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.01,
        zoom_range=0.08,
        fill_mode='nearest',
    )

    # Configure data generator for training masks
    train_mask_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=4,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.01,
        zoom_range=0.08,
        fill_mode='nearest',
    )

    # Configure data generator for validation images
    val_image_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    # Configure data generator for validation masks
    val_mask_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    # Create training data generator from DataFrame
    train_image_generator = train_image_datagen.flow_from_dataframe(
        dataframe=df[df['set'] == 'train'],
        x_col='image_path',
        y_col=None,
        target_size=(params.height, params.width),
        batch_size=params.batch_size,
        class_mode=None,
        seed=params.seed
    )

    train_mask_generator = train_mask_datagen.flow_from_dataframe(
        dataframe=df[df['set'] == 'train'],
        x_col='mask_path',
        y_col=None,
        target_size=(params.height, params.width),
        batch_size=params.batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=params.seed
    )

    # Create validation data generator from DataFrame
    val_image_generator = val_image_datagen.flow_from_dataframe(
        dataframe=df[df['set'] == 'val'],
        x_col='image_path',
        y_col=None,
        target_size=(params.height, params.width),
        batch_size=params.batch_size,
        class_mode=None,
        seed=params.seed
    )

    val_mask_generator = val_mask_datagen.flow_from_dataframe(
        dataframe=df[df['set'] == 'val'],
        x_col='mask_path',
        y_col=None,
        target_size=(params.height, params.width),
        batch_size=params.batch_size,
        class_mode=None,
        color_mode='grayscale',
        seed=params.seed
    )

    # Combine training data generators using zip
    train_generator = zip( train_image_generator, train_mask_generator )

    # Combine validation data generators using zip
    val_generator = zip( val_image_generator, val_mask_generator )

    return train_generator, val_generator


def create_and_train_model(train_generator, val_generator, df):
    """
    Function to create and train a U-Net model using the provided training and validation data generators.

    This function builds a U-Net model, compiles it with custom metrics and loss functions,
    and trains the model using the specified data generators. Model checkpoints and early stopping
    are used to ensure the best model is saved and training stops if no improvement is seen.

    Args:
    train_generator (zip): Generator for training images and masks.
    val_generator (zip): Generator for validation images and masks.
    df (pd.DataFrame): DataFrame containing image and mask paths along with set labels.

    Returns:
    history (History): Training history of the model.
    """
    print( '----Model training--------------------------------------------' )

    # Build the U-Net model with specified input shape
    model = build_unet( (params.height, params.width, params.number_of_channels) )

    # Register custom objects for dice coefficient and dice loss
    tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

    # Compile the model with Adam optimizer, dice loss, and dice coefficient metric
    model.compile( optimizer=tf.keras.optimizers.Adam( learning_rate=params.lr ),
                   loss=[dice_loss],
                   metrics=[dice_coefficient])

    # Define model checkpoint to save the best model based on dice coefficient
    checkpoint = ModelCheckpoint( pathes.model_unet_path, monitor='val_loss', verbose=1, save_best_only=True )

    # Define early stopping to halt training if no improvement in dice coefficient is observed
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True )

    # Calculate steps per epoch for training and validation
    steps_per_epoch = len( df[df['set'] == 'train'] ) // params.batch_size
    validation_steps = len( df[df['set'] == 'val'] ) // params.batch_size

    # Train the model using the training and validation generators
    history = model.fit( train_generator, steps_per_epoch=steps_per_epoch, epochs=params.epochs,
                         validation_data=val_generator, validation_steps=validation_steps,
                         callbacks=[checkpoint, early_stopping] )

    return model


def main():
    mri_mask_df = create_mri_mask_df()
    mri_mask_df.drop('set', axis=1).to_csv(pathes.mri_mask_path)
    train_generator, val_generator = create_generators(mri_mask_df)
    model = create_and_train_model(train_generator, val_generator, mri_mask_df)
if __name__ == "__main__":
    main()
