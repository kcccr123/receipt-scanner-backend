import os
import copy
import typing
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from dataUtils import Transformer, Augmentor

class DataLoader:
    def __init__(self, dataset: typing.Union[str, list, pd.DataFrame], data_preprocessors: typing.List[typing.Callable] = None,
                batch_size: int = 4, initial_epoch: int = 1, augmentors: typing.List[Augmentor] = None,
                transformers: typing.List[Transformer] = None, batch_postprocessors: typing.List[typing.Callable] = None,
                log_level: int = logging.INFO) -> None:
        
        self._dataset = dataset
        self._data_preprocessors = [] if data_preprocessors is None else data_preprocessors
        self._batch_size = batch_size
        self._epoch = initial_epoch
        self._augmentors = [] if augmentors is None else augmentors
        self._transformers = [] if transformers is None else transformers
        self._batch_postprocessors = [] if batch_postprocessors is None else batch_postprocessors
        self._step = 0
        self._cache = {}
        self._on_epoch_end_remove = []

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Check if dataset has length
        if not len(dataset):
            raise ValueError("Dataset must be iterable")

    def __len__(self):
        #number of batches per epoch
        return int(np.ceil(len(self._dataset) / self._batch_size))

    @property
    def augmentors(self) -> typing.List[Augmentor]:
        return self._augmentors

    @augmentors.setter
    def augmentors(self, augmentors: typing.List[Augmentor]):
        for augmentor in augmentors:
            if isinstance(augmentor, Augmentor):
                if self._augmentors is not None:
                    self._augmentors.append(augmentor)
                else:
                    self._augmentors = [augmentor]

            else:
                self.logger.warning(f"Augmentor {augmentor} is not an instance of Augmentor.")

    @property
    def transformers(self) -> typing.List[Transformer]:
        return self._transformers

    @transformers.setter
    def transformers(self, transformers: typing.List[Transformer]):
        for transformer in transformers:
            if isinstance(transformer, Transformer):
                if self._transformers is not None:
                    self._transformers.append(transformer)
                else:
                    self._transformers = [transformer]

            else:
                self.logger.warning(f"Transformer {transformer} is not an instance of Transformer.")

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    def on_epoch_end(self):

        self._epoch += 1 #increment epoch
        #shuffle dataset
        np.random.shuffle(self._dataset)

        # Remove any invalide data
        for remove in self._on_epoch_end_remove:
            self.logger.warning(f"Removing {remove} from dataset.")
            self._dataset.remove(remove)
        self._on_epoch_end_remove = []

    def split(self, split: float = 0.9, shuffle: bool = True) -> typing.Tuple[typing.Any, typing.Any]:

        if shuffle:
            np.random.shuffle(self._dataset)
            
        train_data_loader, val_data_loader = copy.deepcopy(self), copy.deepcopy(self)
        print("done coopying")
        train_data_loader._dataset = self._dataset[:int(len(self._dataset) * split)]
        val_data_loader._dataset = self._dataset[int(len(self._dataset) * split):]

        return train_data_loader, val_data_loader

    def to_csv(self, path: str, index: bool = False) -> None:

        df = pd.DataFrame(self._dataset)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=index)

    def get_batch_annotations(self, index: int) -> typing.List:
        #Returns a batch of annotations by batch index in the dataset
        self._step = index
        start_index = index * self._batch_size

        # Get batch indexes
        batch_indexes = [i for i in range(start_index, start_index + self._batch_size) if i < len(self._dataset)]

        # Read batch data
        batch_annotations = [self._dataset[index] for index in batch_indexes]

        return batch_annotations
    
    def start_executor(self) -> None:
        #Start the executor to process data
        def executor(batch_data):
            for data in batch_data:
                yield self.process_data(data) #use yield to save memory space

        if not hasattr(self, "_executor"):
            self._executor = executor

    def __iter__(self):
        #Create a generator that iterate over the Sequence.
        for index in range(len(self)):
            results = self[index]
            yield results

    def process_data(self, batch_data):
        #Process data batch of data
        if batch_data[0] in self._cache and isinstance(batch_data[0], str):
            data, annotation = copy.deepcopy(self._cache[batch_data[0]])
        else:
            data, annotation = batch_data
            for preprocessor in self._data_preprocessors:
                data, annotation = preprocessor(data, annotation)
            
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, marking for removal on epoch end.")
                self._on_epoch_end_remove.append(batch_data)
                return None, None
            
            if batch_data[0] not in self._cache:
                self._cache[batch_data[0]] = (copy.deepcopy(data), copy.deepcopy(annotation))

        # Then augment, transform and postprocess the batch data
        for tools in [self._augmentors, self._transformers]:
            for change in tools:
                data, annotation = change(data, annotation)

        try:
            data = data.numpy()
            annotation = annotation.numpy()
        except:
            pass

        return data, annotation

    def __getitem__(self, index: int):
        #Returns a batch of processed data by index

        if index==0:
            self.start_executor()

        dataset_batch = self.get_batch_annotations(index)
        
        # First read and preprocess the batch data
        batch_data, batch_annotations = [], []
        for data, annotation in self._executor(dataset_batch):
            if data is None or annotation is None:
                self.logger.warning("Data or annotation is None, skipping.")
                continue
            batch_data.append(data)
            batch_annotations.append(annotation)

        if self._batch_postprocessors:
            for batch_postprocessor in self._batch_postprocessors:
                batch_data, batch_annotations = batch_postprocessor(batch_data, batch_annotations)

            return batch_data, batch_annotations

        try:
            return np.array(batch_data), np.array(batch_annotations)
        except:
            return batch_data, batch_annotations