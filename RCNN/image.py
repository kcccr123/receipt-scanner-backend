import os
import cv2
import typing
import logging
from abc import ABC
from abc import abstractmethod
import numpy as np

class Image(ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def shape(self) -> tuple:
        pass

    @property
    @abstractmethod
    def center(self) -> tuple:
        pass

    @abstractmethod
    def RGB(self) -> np.ndarray:
        pass

    @abstractmethod
    def HSV(self) -> np.ndarray:
        pass

    @abstractmethod
    def update(self, image: np.ndarray):
        pass

    @abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abstractmethod
    def numpy(self) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        pass


class CVImage(Image):
    #Image class for storing image data and metadata (opencv based)

    init_successful = False

    def __init__(self, image: typing.Union[str, np.ndarray], method: int = cv2.IMREAD_COLOR,
                path: str = "", color: str = "BGR") -> None:
        super().__init__()
        
        if isinstance(image, str):#image path
            if not os.path.exists(image):
                raise FileNotFoundError(f"Image {image} not found.")

            self._image = cv2.imread(image, method)
            self.path = image
            self.color = "BGR"

        elif isinstance(image, np.ndarray):
            self._image = image
            self.path = path
            self.color = color

        else:
            raise TypeError(f"Image must be either path to image or numpy.ndarray, not {type(image)}")

        self.method = method

        if self._image is None:
            return None

        self.init_successful = True

        # save width, height and channels
        self.width = self._image.shape[1]
        self.height = self._image.shape[0]
        self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

    @property
    def image(self) -> np.ndarray:
        return self._image

    @image.setter
    def image(self, value: np.ndarray):
        self._image = value

    @property
    def shape(self) -> tuple:
        return self._image.shape

    @property
    def center(self) -> tuple:
        return self.width // 2, self.height // 2

    def RGB(self) -> np.ndarray:
        if self.color == "RGB":
            return self._image
        elif self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unknown color format {self.color}")
        
    def HSV(self) -> np.ndarray:
        if self.color == "BGR":
            return cv2.cvtColor(self._image, cv2.COLOR_BGR2HSV)
        elif self.color == "RGB":
            return cv2.cvtColor(self._image, cv2.COLOR_RGB2HSV)
        else:
            raise ValueError(f"Unknown color format {self.color}")

    def update(self, image: np.ndarray):
        if isinstance(image, np.ndarray):
            self._image = image

            # save width, height and channels
            self.width = self._image.shape[1]
            self.height = self._image.shape[0]
            self.channels = 1 if len(self._image.shape) == 2 else self._image.shape[2]

            return self

        else:
            raise TypeError(f"image must be numpy.ndarray, not {type(image)}")

    def flip(self, axis: int = 0):
        #flip image along x or y axis
        if axis not in [0, 1]:
            raise ValueError(f"axis must be either 0 or 1, not {axis}")

        self._image = self._image[:, ::-1] if axis == 0 else self._image[::-1]

        return self

    def numpy(self) -> np.ndarray:
        return self._image
    
    def __call__(self) -> np.ndarray:
        return self._image

class ImageReader:
    #extract image and label from path
    def __init__(self, image_class: Image, log_level: int = logging.INFO, ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self._image_class = image_class

    def __call__(self, image_path: typing.Union[str, np.ndarray], label: typing.Any) -> typing.Tuple[Image, typing.Any]:

        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image {image_path} not found.")
        elif isinstance(image_path, np.ndarray):
            pass
        else:
            raise TypeError(f"Image {image_path} is not a string or numpy array.")

        image = self._image_class(image=image_path)

        if not image.init_successful:
            image = None
            self.logger.warning(f"Image {image_path} could not be read, returning None.")

        return image, label


