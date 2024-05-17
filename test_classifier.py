import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

import path
import columns
import pickle

from tensorflow.keras.models import load_model

from train_unet import dice_coefficient, dice_loss
from train_classifier import combine_tabular_with_feature_map

def preprocess(df_original):
    """
        Function to preprocess the DataFrame for model testing.

        This function performs several preprocessing steps, including dropping
        specified columns, handling missing values through various imputation
        methods, frequency encoding of categorical variables, and normalization
        of numeric features.

        Args:
        df_original (pd.DataFrame): Original DataFrame containing the raw data.

        Returns:
        df (pd.DataFrame): Preprocessed DataFrame ready for model testing.
        """
    print('----Preprocessing--------------------------------------------')
    df = df_original.copy()
    df.drop('tumor_tissue_site', axis=1, inplace=True)
    param_dict = pickle.load(open(path.param_dict, 'rb'))

    for col in columns.numeric_cols:
        df.fillna({col: param_dict['mean_impute_values'][col]},inplace=True)

    for col in columns.mode_impute_cols:
        df.fillna({col: param_dict['mode_impute_values'][col]}, inplace=True)

    for col in columns.minus_one_impute:
        df.fillna({col: -1}, inplace=True)

    for col in columns.freq_cols:
        df[col + '_freq'] = df[col].map(param_dict['freq_encode_values'][col + '_freq'])

    scaler = MinMaxScaler()

    columns_for_normalization = columns.numeric_cols

    df[columns_for_normalization] = scaler.fit_transform( df[columns_for_normalization] )

    print('Preprocessed df info: ')
    print(df.info())
    return df


def predict_death(df, test_df):
    """
    Function to predict death using a trained classifier model.

    This function takes tabular data, preprocesses it, loads a pre-trained classifier model,
    predicts death probabilities for the test data, and adds the predictions to the test DataFrame.

    Args:
    df (pd.DataFrame): DataFrame containing tabular data for training.
    test_df (pd.DataFrame): DataFrame containing tabular data for testing.

    Returns:
    test_df (pd.DataFrame): DataFrame with death predictions added.
    """
    print( '----Predicting--------------------------------------------' )

    X = df.drop( columns.cols_to_drop, axis=1 )

    # Load the pre-trained classifier model
    with open( path.model_classifier_path, 'rb' ) as f:
        model = pickle.load( f )

    if 'death01' in test_df.columns:
        test_df = test_df.drop( 'death01', axis=1 )

    # Predict death 0/1
    test_df['death01'] = model.predict(X)

    return test_df


def main():
    tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
    tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

    test_df = pd.read_csv(path.test_csv_path)
    changed_df = preprocess(test_df)
    model_unet = load_model(path.model_unet_path)
    mri_mask_df = pd.read_csv(path.mri_mask_path)
    # tabular_and_feature_df = pd.read_csv(path.tabular_and_feature_df_path_test)
    tabular_and_feature_df = combine_tabular_with_feature_map(model_unet, changed_df, mri_mask_df)
    tabular_and_feature_df.to_csv(path.tabular_and_feature_df_path_test, index=False)
    predicted_csv = predict_death(tabular_and_feature_df, test_df)
    predicted_csv.to_csv(path.predicted_csv_path, index=False)
    print(f"----Result of predicting saved to {path.predicted_csv_path}----")

if __name__ == "__main__":
    main()