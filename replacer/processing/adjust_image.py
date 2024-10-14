import numpy as np
from PIL.Image import Image
from PIL import ImageEnhance


def adjust_image(
    image: Image, brightness=1.0, contrast=1.0, saturation=1.0, hue=1.0
) -> Image:
    """
    Adjust the brightness, contrast, saturation, and hue of the given image.

    Parameters
    ----------
    image : Image
        The image to be adjusted.
    brightness : float, optional
        The factor by which to adjust the brightness. Default is 1.0.
    contrast : float, optional
        The factor by which to adjust the contrast. Default is 1.0.
    saturation : float, optional
        The factor by which to adjust the saturation. Default is 1.0.
    hue : float, optional
        The factor by which to adjust the hue. Default is 1.0.

    Returns
    -------
    Image
        The adjusted image.
    """
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(saturation)

    if hue != 1.0:
        image = adjust_hue(image, hue)

    return image


def adjust_hue(image: Image, hue_factor: float) -> Image:
    """
    Adjusts the hue of an image by converting to HSV, modifying the hue, and converting back to RGB.

    Parameters
    ----------
    image : Image
        The input image.
    hue_factor : float
        The factor by which to adjust the hue.

    Returns
    -------
    Image
        The adjusted image.
    """
    hsv_image = image.convert("HSV")
    np_image = np.array(hsv_image)
    np_image[:, :, 0] = np_image[:, :, 0] * hue_factor % 256
    adjusted_image = Image.fromarray(np_image, "HSV").convert("RGB")
    return adjusted_image
