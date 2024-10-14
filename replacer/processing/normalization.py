import cv2
import numpy as np
from PIL import Image


def apply_clahe(image: np.ndarray) -> Image.Image:
    """
    Apply Contrast Limited Adaptive Histogram Equalization to an image.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_image[:, :, 0] = clahe.apply(lab_image[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR))


def apply_hist(image: np.ndarray) -> Image.Image:
    """
    Apply histogram equalization to the Y channel of a YUV image.
    """
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
    return Image.fromarray(cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR))


def apply_white_balance(image: np.ndarray) -> Image.Image:
    """
    Adjust white balance by scaling the red and blue channels relative to green.
    """
    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(result)
    l = cv2.equalizeHist(l)
    result = cv2.merge([l, a, b])
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_LAB2BGR))


def apply_gamma_correction(image: np.ndarray, gamma=1.0) -> Image.Image:
    """
    Apply gamma correction to adjust image brightness. Gamma values less than 1.0 make the image darker,
    while values greater than 1.0 make the image brighter.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype(
        "uint8"
    )
    return Image.fromarray(cv2.LUT(image, table))


def normalize_colors(image: Image.Image, method: str = "clahe") -> Image.Image:
    """
    Normalize an image using the specified method.

    Parameters
    ----------
    image : Image.Image
        The input image to be normalized.
    method : str, optional
        The normalization method to use. Options are 'hist', 'clahe', 'white_balance',
        and 'gamma_correction'. Default is 'clahe'.

    Returns
    -------
    Image.Image
        The normalized image.

    Raises
    ------
    ValueError
        If an invalid method is specified.
    """
    image = np.array(image)
    if method == "hist":
        normalized_image = apply_hist(image)
    elif method == "clahe":
        normalized_image = apply_clahe(image)
    elif method == "white_balance":
        normalized_image = apply_white_balance(image)
    elif method == "gamma_correction":
        normalized_image = apply_gamma_correction(image, gamma=0.9)
    else:
        raise ValueError(
            f"Invalid method: {method}. Choose from 'hist', 'clahe', 'white_balance', or 'gamma_correction'."
        )

    return normalized_image
