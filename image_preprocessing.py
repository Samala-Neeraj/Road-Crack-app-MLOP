from os import PathLike
from typing import BinaryIO

from PIL import Image


ImageSource = str | PathLike[str] | bytes | bytearray | BinaryIO


def preprocess_image(source: ImageSource) -> Image.Image:
    image_source: str | PathLike[str] | BinaryIO
    if isinstance(source, (bytes, bytearray)):
        from io import BytesIO

        image_source = BytesIO(source)
    else:
        image_source = source

    with Image.open(image_source) as image:
        image.verify()

    if hasattr(image_source, "seek"):
        image_source.seek(0)
    with Image.open(image_source) as image:
        image.load()
        return image.convert("RGB")
