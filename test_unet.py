import path
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from train_unet import dice_loss, dice_coefficient

def display_image_and_mask(image_path, mask):
    """
    Function to display an original image and its corresponding segmented mask.

    Parameters:
        image_path (str): The file path of the original image.
        mask (numpy.ndarray): The segmented mask image.

    Returns:
        None

    """
    # Read the original image

    # mask = np.expand_dims(mask, axis=-1)
    print(mask.shape)
    mask = np.array(mask * 255).astype('uint8')
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR )
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV reads images in BGR format, so convert it to RGB
    # image = cv2.resize(image, (mask.shape[1], mask.shape[0]))  # Resize image to match mask dimensions
    # Concatenate the original image and the mask horizontally
    combined_image = np.hstack( (image, mask) )


    # Display the combined image
    cv2.imshow('Image and mask', combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

tf.keras.utils.get_custom_objects()['dice_coefficient'] = dice_coefficient
tf.keras.utils.get_custom_objects()['dice_loss'] = dice_loss

model_large = load_model(path.model_unet_path)
image = cv2.imread(path.image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # OpenCV reads images in BGR format, so convert it to RGB
image = image / 255.
image = np.expand_dims(image, axis=0)  # Add a batch dimension
predictions = model_large.predict(np.array(image))
threshold = np.mean(predictions.squeeze())
# Apply the threshold to the predictions
mask = np.where(predictions.squeeze() > threshold, 1, 0)

display_image_and_mask(path.image_path, mask)


