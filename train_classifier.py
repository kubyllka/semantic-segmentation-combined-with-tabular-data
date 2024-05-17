import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import params
import path
import columns
import pickle
import params_xgb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer
from train_unet import dice_coefficient, dice_loss


def train_model(df):
    """
    Function to train an XGBoost classifier using the provided DataFrame.

    This function prepares the feature matrix and target vector, initializes
    and fits an XGBoost classifier, performs cross-validation to evaluate the model,
    and prints the cross-validated metrics.

    Args:
    df (pd.DataFrame): DataFrame containing the training data.

    Returns:
    model (XGBClassifier): Trained XGBoost classifier.
    """
    # Prepare feature matrix X by dropping specified columns
    X = df.drop( columns.cols_to_drop, axis=1 )

    y = df[columns.y]

    # Initialize XGBoost classifier with specified parameters
    model = xgb.XGBClassifier( **params_xgb.params )

    model.fit( X, y )

    # Define scoring metrics for cross-validation
    scoring = {'accuracy': make_scorer( accuracy_score ),
               'precision': make_scorer( precision_score ),
               'recall': make_scorer( recall_score ),
               'f1': make_scorer( f1_score )}

    # Initialize Stratified K-Fold cross-validator
    cv = StratifiedKFold( n_splits=params.num_of_splits, shuffle=True, random_state=params.seed)

    # Perform cross-validation and compute scores
    scores = cross_validate( model, X, y, cv=cv, scoring=scoring)

    print( "Available scores:", scores.keys() )
    print( "Cross-validated Accuracy:", scores['test_accuracy'].mean() )
    print( "Cross-validated Precision:", scores['test_precision'].mean() )
    print( "Cross-validated Recall:", scores['test_recall'].mean() )
    print( "Cross-validated F1 Score:", scores['test_f1'].mean() )

    return model


def extract_features_unet(unet_model, data):
    """
    Extract features using UNet model.

    Args:
        unet_model (tensorflow.keras.Model): UNet model.
        data (numpy.ndarray): Input data.

    Returns:
        numpy.ndarray: Extracted features.
    """
    features_extractor = tf.keras.Model(inputs=unet_model.input, outputs=unet_model.get_layer('conv2d_18').output)
    features = features_extractor.predict(data)
    return features


def preprocess(df_original):
    """
    Function to preprocess the DataFrame for model training.

    This function performs several preprocessing steps, including dropping
    specified columns, handling missing values through various imputation
    methods, frequency encoding of categorical variables, and normalization
    of numeric features. It also saves the parameters used for imputation
    and encoding.

    Args:
    df_original (pd.DataFrame): Original DataFrame containing the raw data.

    Returns:
    df (pd.DataFrame): Preprocessed DataFrame ready for model training.
    """
    print( '----Preprocessing--------------------------------------------' )

    df = df_original.copy()

    # Drop the 'tumor_tissue_site' column
    df.drop( 'tumor_tissue_site', axis=1, inplace=True )

    # Drop rows where the 'death01' column has missing values
    df.dropna( subset=['death01'], inplace=True )

    mean_impute_values = dict()
    # Impute missing values in numeric columns with the mean value
    for col in columns.numeric_cols:
        mean = df[col].mean()
        df.fillna( {col: mean}, inplace=True )
        mean_impute_values[col] = mean

    mode_impute_values = dict()
    # Impute missing values in specified columns with the mode value
    for col in columns.mode_impute_cols:
        mode = df[col].mode()[0]
        df.fillna( {col: mode}, inplace=True )
        mode_impute_values[col] = mode

    # Impute missing values in specified columns with -1
    for col in columns.minus_one_impute:
        df.fillna( {col: -1}, inplace=True )

    freq_encode_values = dict()

    # Frequency encode specified columns
    for col in columns.freq_cols:
        frequency_encoding = df[col].value_counts( normalize=True )
        df[col + '_freq'] = df[col].map( frequency_encoding )
        freq_encode_values[col + '_freq'] = df[col].map( frequency_encoding )


    # Save imputation and encoding parameters for future use
    param_dict = {
        'mean_impute_values': mean_impute_values,
        'mode_impute_values': mode_impute_values,
        'freq_encode_values': freq_encode_values
    }

    scaler = MinMaxScaler()
    columns_for_normalization = columns.numeric_cols

    # Normalize the specified numeric columns
    df[columns_for_normalization] = scaler.fit_transform( df[columns_for_normalization] )

    with open( path.param_dict, 'wb' ) as handle:
        pickle.dump( param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL )

    return df


def combine_tabular_with_feature_map(model, df, mri_mask_df):
    """
    Function to combine tabular data with feature maps extracted from MRI images.

    This function extracts feature maps using a pre-trained model from MRI images
    and combines them with tabular data for each patient.

    Args:
    model (tf.keras.Model): Pre-trained U-Net model for feature extraction.
    df (pd.DataFrame): DataFrame containing tabular data.
    mri_mask_df (pd.DataFrame): DataFrame containing MRI image paths and corresponding patient IDs.

    Returns:
    df_with_feature_map (pd.DataFrame): DataFrame containing tabular data combined with feature maps.
    """
    print( '----Extract feature data and combine with tabular data--------------------------------------------' )


    df_with_feature_map = pd.DataFrame()

    # Data generator for MRI images
    mri_datagen = ImageDataGenerator(
        rescale=1. / 255,
    )

    for idx, row in df.iterrows():
        # Get tabular data for the current patient
        df_data_info = df[df['Patient'] == row['Patient']]
        # Get MRI data for the current patient
        df_patient_mri = mri_mask_df[mri_mask_df['patient_id'] == row['Patient']]

        # Create a generator for MRI images of the current patient
        mri_generator = mri_datagen.flow_from_dataframe(
            dataframe=df_patient_mri,
            x_col='image_path',
            y_col=None,
            target_size=(params.height, params.width),
            batch_size=params.batch_size,
            class_mode=None,
            shuffle=True,
            seed=params.seed
        )

        # Extract features from MRI images using the pre-trained model
        features_unet_patient = extract_features_unet( model, mri_generator )

        # Compute mean values of the feature maps
        mean_values = np.mean( features_unet_patient, axis=(1, 2, 3) )
        max_index = np.argmax( mean_values )

        # Select the feature map with the highest mean value
        selected_feature_map = features_unet_patient[max_index]

        # Reshape the selected feature map to a vector
        features_unet_vectorized = selected_feature_map.reshape( 1, -1 )

        # Create a DataFrame with the feature map for the current patient
        df_patient_feature_map = pd.DataFrame( data=features_unet_vectorized, columns=[f'feature_{i}' for i in range(
            features_unet_vectorized.shape[1] )] )

        # Combine tabular data and feature map data for the current patient
        df_patient_data = pd.concat( [df_data_info.reset_index( drop=True ), df_patient_feature_map], axis=1 )

        # Concatenate patient data with the overall DataFrame
        df_with_feature_map = pd.concat( [df_with_feature_map, df_patient_data], ignore_index=True )

    return df_with_feature_map


def main():
    tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss
    train_df = pd.read_csv(path.train_csv_path)
    changed_df = preprocess(train_df)
    model_unet = load_model(path.model_unet_path)
    mri_mask_df = pd.read_csv(path.mri_mask_path)
    tabular_and_feature_df = combine_tabular_with_feature_map(model_unet, changed_df, mri_mask_df)
    tabular_and_feature_df.to_csv(path.tabular_and_feature_df_path, index=False)
    tabular_and_feature_df = pd.read_csv(path.tabular_and_feature_df_path)
    trained_model = train_model(tabular_and_feature_df)
    with open(path.model_classifier_path, 'wb' ) as file:
        pickle.dump(trained_model, file )


if __name__ == "__main__":
    main()
