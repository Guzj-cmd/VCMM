"""RandAugment operations used by the released image--text example."""

import cv2
import numpy as np


def _identity(image):
    return image


def _autocontrast(image, cutoff=0):
    bins = 256

    def tune(channel):
        count = channel.size
        cut = cutoff * count // 100
        if cut == 0:
            high, low = channel.max(), channel.min()
        else:
            histogram = cv2.calcHist([channel], [0], None, [bins], [0, bins])
            lows = np.argwhere(np.cumsum(histogram) > cut)
            low = 0 if lows.shape[0] == 0 else lows[0]
            highs = np.argwhere(np.cumsum(histogram[::-1]) > cut)
            high = bins - 1 if highs.shape[0] == 0 else bins - 1 - highs[0]
        if high <= low:
            table = np.arange(bins)
        else:
            scale = (bins - 1) / (high - low)
            offset = -low * scale
            table = np.arange(bins) * scale + offset
            table[table < 0] = 0
            table[table > bins - 1] = bins - 1
        table = table.clip(0, 255).astype(np.uint8)
        return table[channel]

    return cv2.merge([tune(channel) for channel in cv2.split(image)])


def _equalize(image):
    def tune(channel):
        histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
        nonzero = histogram[histogram != 0].reshape(-1)
        step = np.sum(nonzero[:-1]) // 255
        if step == 0:
            return channel
        values = np.empty_like(histogram)
        values[0] = step // 2
        values[1:] = histogram[:-1]
        table = (np.cumsum(values) // step).clip(0, 255).astype(np.uint8)
        return table[channel]

    return cv2.merge([tune(channel) for channel in cv2.split(image)])


def _brightness(image, factor):
    table = (np.arange(256, dtype=np.float32) * factor).clip(0, 255)
    return table.astype(np.uint8)[image]


def _sharpness(image, factor):
    kernel = np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 5
    kernel /= 13
    blurred = cv2.filter2D(image, -1, kernel)
    if factor == 0.0:
        return blurred
    if factor == 1.0:
        return image
    output = image.astype(np.float32)
    blurred = blurred.astype(np.float32)[1:-1, 1:-1]
    output[1:-1, 1:-1] = blurred + factor * (
        output[1:-1, 1:-1] - blurred
    )
    return output.astype(np.uint8)


def _rotate(image, degrees, fill=(128, 128, 128)):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1)
    return cv2.warpAffine(image, matrix, (width, height), borderValue=fill)


def _shear_x(image, factor, fill=(128, 128, 128)):
    height, width = image.shape[:2]
    matrix = np.float32([[1, factor, 0], [0, 1, 0]])
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)


def _shear_y(image, factor, fill=(128, 128, 128)):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, 0], [factor, 1, 0]])
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)


def _translate_x(image, offset, fill=(128, 128, 128)):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, -offset], [0, 1, 0]])
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)


def _translate_y(image, offset, fill=(128, 128, 128)):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, 0], [0, 1, -offset]])
    return cv2.warpAffine(
        image, matrix, (width, height), borderValue=fill, flags=cv2.INTER_LINEAR
    ).astype(np.uint8)


_OPERATIONS = {
    "Identity": _identity,
    "AutoContrast": _autocontrast,
    "Equalize": _equalize,
    "Brightness": _brightness,
    "Sharpness": _sharpness,
    "ShearX": _shear_x,
    "ShearY": _shear_y,
    "TranslateX": _translate_x,
    "TranslateY": _translate_y,
    "Rotate": _rotate,
}


def _arguments(name, level):
    if name in {"Identity", "AutoContrast", "Equalize"}:
        return ()
    if name in {"Brightness", "Sharpness"}:
        return ((level / 10) * 1.8 + 0.1,)
    if name in {"ShearX", "ShearY"}:
        value = (level / 10) * 0.3
        if np.random.random() > 0.5:
            value = -value
        return (value, (128, 128, 128))
    if name in {"TranslateX", "TranslateY"}:
        value = (level / 10) * 10
        if np.random.random() > 0.5:
            value = -value
        return (value, (128, 128, 128))
    degrees = (level / 10) * 30
    if np.random.random() < 0.5:
        degrees = -degrees
    return (degrees, (128, 128, 128))


class ImageRandomAugment:
    OPERATIONS = tuple(_OPERATIONS)

    def __init__(self, num_ops=2, magnitude=7):
        self.num_ops = num_ops
        self.magnitude = magnitude

    def __call__(self, image):
        image = np.array(image)
        names = np.random.choice(self.OPERATIONS, self.num_ops)
        for name in names:
            if np.random.random() <= 0.5:
                image = _OPERATIONS[name](
                    image, *_arguments(name, self.magnitude)
                )
        return image
