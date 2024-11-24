import os
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


