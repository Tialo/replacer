import numpy as np
from PIL import Image
from rembg import remove as _remove
from rembg.sessions import sessions_names
from rembg.session_factory import new_session


def find_bounding_box(image: np.ndarray) -> tuple[int, int, int, int]:
    """
    Identify the bounding box of non-zero pixels in an image.

    Parameters
    ----------
    image : np.ndarray
        An array representing the image.

    Returns
    -------
    tuple[int, int, int, int]
        A tuple containing the coordinates of the bounding box in the format (top, bottom, left, right).
    """
    non_zero_pixels = np.any(image != 0, axis=-1)

    rows = np.any(non_zero_pixels, axis=1)
    cols = np.any(non_zero_pixels, axis=0)

    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(np.flip(rows))
    left = np.argmax(cols)
    right = len(cols) - np.argmax(np.flip(cols))

    return top, bottom, left, right


def center_bounding_box(image: np.ndarray) -> np.ndarray:
    """
    Center the non-transparent pixels in the given image.

    Parameters
    ----------
    image : np.ndarray
        The input image.

    Returns
    -------
    np.ndarray
        New centered image.
    """
    top, bottom, left, right = find_bounding_box(image)

    bbox_image = image[top:bottom, left:right]

    img_height, img_width, _ = image.shape
    bbox_height, bbox_width, _ = bbox_image.shape

    new_top = (img_height - bbox_height) // 2
    new_bottom = new_top + bbox_height
    new_left = (img_width - bbox_width) // 2
    new_right = new_left + bbox_width

    centered_image = np.zeros_like(image)

    centered_image[new_top:new_bottom, new_left:new_right] = bbox_image

    return centered_image


def remove(
    input_image: Image.Image, center: bool = False, model: str | None = None
) -> Image.Image:
    """
    Remove the background from the given input image.

    Parameters
    ----------
    input_image : Image.Image
        The input image from which the background will be removed.
    center : bool, optional
        If True, centers the bounding box of the resulting image. Default is False.
    model : str or None, optional
        The name of the model to use for background removal. If None, a default model is used.
        Used in rembg library.

    Returns
    -------
    Image.Image
        The image with the background removed.
    """
    if model is not None:
        assert (
            model in sessions_names
        ), f"Invalid model name. Choose from {sessions_names}"
        model = new_session(model)

    output_image = _remove(input_image, session=model)

    if center:
        output_image = np.array(output_image)
        output_image = Image.fromarray(center_bounding_box(output_image))

    return output_image
