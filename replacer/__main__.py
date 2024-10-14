import argparse

from PIL import Image

from .replacer import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Process input and output image paths."
    )
    parser.add_argument("input_path", type=str, help="Path to the input image file")
    parser.add_argument("output_path", type=str, help="Path to the output image file")

    args = parser.parse_args()

    input_image = Image.open(args.input_path)
    output_image = pipeline(input_image)
    output_image.save(args.output_path)


if __name__ == "__main__":
    main()
