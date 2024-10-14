import logging

import torch
from PIL import Image
from diffusers import AutoPipelineForText2Image
from transformers import CLIPProcessor, CLIPModel

from .prompts import _default_prompts
from .processing.remove_background import remove
from .processing.normalization import normalize_colors
from .processing.adjust_image import adjust_image

logger = logging.getLogger(__name__)

_clip_preprocessor = None
_clip_model = None
_bg_model = None


def _load_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model
def _load_clip_preprocessor():
    global _clip_preprocessor
    if _clip_preprocessor is None:
        _clip_preprocessor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_preprocessor


def _load_bg_model():
    global _bg_model
    if _bg_model is None:
        _bg_model = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")
    return _bg_model


def classify_object(image: Image, labels: list[str] = None) -> str:
    """
    Return the most appropriate label for an image.

    Parameters
    ----------
    image : Image
        The image to be classified.
    labels : list of str, optional
        A list of labels to classify the image against. If not provided, default labels will be used.

    Returns
    -------
    str
        The label that best fits the image.
    """
    if labels is None:
        labels = list(_default_prompts)

    processor = _load_clip_preprocessor()
    model = _load_clip_model()
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    return labels[outputs.logits_per_image.argmax()]


def create_background(prompt: str) -> Image:
    """
    Generate background by given prompt.

    Parameters
    ----------
    prompt : str
        The text prompt used to generate the background image.

    Returns
    -------
    Image
        The generated background image.
    """
    pipe = _load_bg_model()
    return pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]


def replace_background(
    foreground_image: Image.Image, background_image: Image.Image
) -> Image.Image:
    """
    Replace the transparent pixels in the foreground image (those with RGBA values [0, 0, 0, 0])
    with the corresponding pixels from the background image.

    Parameters
    ----------
        foreground_image : Image.Image
          The foreground image with potential transparent areas.
        background_image : Image.Image
          The background image to use for replacing transparent pixels.

    Returns
    -------
        Image.Image: The resulting image with the background pixels replacing the transparent areas of the foreground image.
    """
    background_image = background_image.resize(foreground_image.size).convert("RGBA")
    return Image.alpha_composite(background_image, foreground_image)


def pipeline(
    input_image: Image.Image, return_intermediate: bool = False
) -> Image.Image | list[Image.Image]:
    """
    Generate a new image by replacing the background of the input image with a new background that fits to its category.

    Parameters
    ----------
    input_image : Image.Image
        The input image to be processed.
    return_intermediate : bool, optional
        If True, returns a list of intermediate images at each stage of the pipeline. Defaults to False.

    Returns
    -------
    Image.Image or list of Image.Image
        The final processed image, or a list of intermediate images if return_intermediate is True.
    """
    logger.info("Starting pipeline processing.")
    logger.debug(f"Input image size: {input_image.size}")

    foreground_image = remove(input_image, center=True, model="birefnet-general")
    logger.info("Foreground image extracted.")
    logger.debug(f"Foreground image size: {foreground_image.size}")

    label = classify_object(input_image)  # it's better to classify the original image
    logger.info(f"Image classified as: {label}")

    prompt = _default_prompts[label]
    logger.debug(f"Using prompt: {prompt}")

    background_image = create_background(prompt)
    logger.info("Background image created.")
    logger.debug(f"Background image size: {background_image.size}")

    new_image = replace_background(foreground_image, background_image)
    logger.info("Background replaced.")

    normalized_image = normalize_colors(new_image)
    logger.info("Image colors normalized.")

    final_image = adjust_image(
        normalized_image, brightness=0.8, contrast=0.95, saturation=1.5
    )

    if return_intermediate:
        return [
            foreground_image,
            background_image,
            new_image,
            normalized_image,
            final_image,
        ]
    return final_image
