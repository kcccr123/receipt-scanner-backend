import cv2
import typing
import numpy as np
import importlib
from image import Image


def randomness(func):

    def wrapper(self, data: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:

        if not isinstance(data, Image):
            self.logger.error(f"data must be Image object, not {type(data)}, skipping augmentor")
            return data, annotation
        
        if np.random.rand() > self._random_chance:
            return data, annotation #return unaugmented image

        # Augment Image
        return func(self, data, annotation)

    return wrapper

class Augmentor:
    #base augmentor
    def __init__(self, random_chance: float=0.5) -> None:
        
        if random_chance > 1:
            self._random_chance = 1
        elif random_chance <0:
            self._random_chance = 0
        else:
            self._random_chance = random_chance #0.0 is never and 1.0 is always.

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    def augment(self, data: Image):
        raise NotImplementedError

    @randomness
    def __call__(self, data: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:

        data = self.augment(data)

        return data, annotation


class RandomBrightness(Augmentor):

    def __init__(self, random_chance: float = 0.5, delta: int = 100) -> None:

        super(RandomBrightness, self).__init__(random_chance)

        assert 0 <= delta <= 255.0, "Delta must be between 0.0 and 255.0"

        self._delta = delta

    def augment(self, image: Image, value: float) -> Image:

        hsv = np.array(image.HSV(), dtype = np.float32)

        hsv[:, :, 1] = hsv[:, :, 1] * value
        hsv[:, :, 2] = hsv[:, :, 2] * value

        hsv = np.uint8(np.clip(hsv, 0, 255))

        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        image.update(img)

        return image

    @randomness
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:

        value = 1 + np.random.uniform(-self._delta, self._delta) / 255

        image = self.augment(image, value)
        return image, annotation

class RandomRotate(Augmentor):

    def __init__(self, random_chance: float = 0.5, angle: typing.Union[int, typing.List]=30, 
                borderValue: typing.Tuple[int, int, int]=None) -> None:

        super(RandomRotate, self).__init__(random_chance)

        self._angle = angle
        self._borderValue = borderValue

    @staticmethod
    def rotate_image(image: np.ndarray, angle: typing.Union[float, int], borderValue: tuple=(0,0,0), return_rotation_matrix: bool=False) -> np.ndarray:
        # grab the dimensions of the image and then determine the centre
        height, width = image.shape[:2]
        center_x, center_y = (width // 2, height // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((height * sin) + (width * cos))
        nH = int((height * cos) + (width * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - center_x
        M[1, 2] += (nH / 2) - center_y

        # rotate
        img = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue)

        if return_rotation_matrix:
            return img, M
        
        return img

    @randomness
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:

        # check if angle is list of angles or a single angle value
        if isinstance(self._angle, list):
            angle = float(np.random.choice(self._angle))
        else:
            angle = float(np.random.uniform(-self._angle, self._angle))

        # generate random border color
        borderValue = np.random.randint(0, 255, 3) if self._borderValue is None else self._borderValue
        borderValue = [int(v) for v in borderValue]

        img, rotMat = self.rotate_image(image.numpy(), angle, borderValue, return_rotation_matrix=True)


        if isinstance(annotation, Image):
            # perform the actual rotation and return the annotation image
            annotation_image = self.rotate_image(annotation.numpy(), angle, borderValue=(0, 0, 0))
            annotation.update(annotation_image)

        image.update(img)

        return image, annotation


class RandomErodeDilate(Augmentor):
    def __init__(self, random_chance: float = 0.5, kernel_size: typing.Tuple[int, int]=(1, 1)) -> None:
       
        super(RandomErodeDilate, self).__init__(random_chance)
        self._kernel_size = kernel_size
        self.kernel = np.ones(self._kernel_size, np.uint8)

    def augment(self, image: Image) -> Image:
        if np.random.rand() <= 0.5:
            img = cv2.erode(image.numpy(), self.kernel, iterations=1)
        else:
            img = cv2.dilate(image.numpy(), self.kernel, iterations=1)

        image.update(img)

        return image

    @randomness
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        image = self.augment(image)

        return image, annotation


class RandomSharpen(Augmentor):
    def __init__(self, random_chance: float = 0.5, alpha: float = 0.25, lightness_range: typing.Tuple = (0.75, 2.0),
                kernel: np.ndarray = None, kernel_anchor: np.ndarray = None) -> None:
       
        super(RandomSharpen, self).__init__(random_chance)

        self._alpha_range = (alpha, 1.0)
        self._ligtness_range = lightness_range
        self._lightness_anchor = 8

        self._kernel = np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1, -1]], dtype=np.float32) if kernel is None else kernel
        self._kernel_anchor = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32) if kernel_anchor is None else kernel_anchor

        assert 0 <= alpha <= 1.0, "Alpha must be between 0.0 and 1.0"

    def augment(self, image: Image) -> Image:
        lightness = np.random.uniform(*self._ligtness_range)
        alpha = np.random.uniform(*self._alpha_range)

        kernel = self._kernel_anchor  * (self._lightness_anchor + lightness) + self._kernel
        kernel -= self._kernel_anchor
        kernel = (1 - alpha) * self._kernel_anchor + alpha * kernel

        # Apply sharpening to each channel
        r, g, b = cv2.split(image.numpy())
        r_sharp = cv2.filter2D(r, -1, kernel)
        g_sharp = cv2.filter2D(g, -1, kernel)
        b_sharp = cv2.filter2D(b, -1, kernel)

        # Merge the sharpened channels back into the original image
        image.update(cv2.merge([r_sharp, g_sharp, b_sharp]))

        return image

    @randomness
    def __call__(self, image: Image, annotation: typing.Any) -> typing.Tuple[Image, typing.Any]:
        image = self.augment(image)
        return image, annotation

class Transformer:
    def __init__(self) -> None:
        pass

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError
    
class ImageResizer(Transformer):
    #doesn't keep aspect ratio
    def __init__(self, width: int, height: int, padding_color: typing.Tuple[int]=(0, 0, 0)) -> None:
        self._width = width
        self._height = height
        self._padding_color = padding_color

    @staticmethod
    def unpad_maintaining_aspect_ratio(padded_image: np.ndarray, original_width: int, original_height: int) -> np.ndarray:
        height, width = padded_image.shape[:2]
        ratio = min(width / original_width, height / original_height)

        delta_w = width - int(original_width * ratio)
        delta_h = height - int(original_height * ratio)
        left, right = delta_w//2, delta_w-(delta_w//2)
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        unpaded_image = padded_image[top:height-bottom, left:width-right]

        original_image = cv2.resize(unpaded_image, (original_width, original_height))

        return original_image

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        
        if not isinstance(image, Image):
            raise TypeError(f"Expected image to be of type Image, got {type(image)}")

        image_numpy = cv2.resize(image.numpy(), (self._width, self._height))
        if isinstance(label, Image):
            label_numpy = cv2.resize(label.numpy(), (self._width, self._height))
            label.update(label_numpy)

        image.update(image_numpy)

        return image, label
    
class LabelIndexer(Transformer):
    #Convert label to index by vocab
    
    def __init__(self, vocab: typing.List[str]) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class LabelPadding(Transformer):
    #Pad label to max_word_length
    def __init__(self, padding_value: int, max_word_length: int = None) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value

    def __call__(self, data: np.ndarray, label: np.ndarray):

        label = label[:self.max_word_length]
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)


