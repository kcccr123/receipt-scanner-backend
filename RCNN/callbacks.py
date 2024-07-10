import os
import onnx
import logging
import typing
import numpy as np
from pathlib import Path
from datetime import datetime
import torch.onnx
from torch.utils.tensorboard import SummaryWriter
import logging

#base class
class Callback:
    def __init__(
        self, 
        monitor: str = "val_loss"
    ) -> None:
        self.monitor = monitor
        logging.basicConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_batch_begin(self, batch: int, logs=None):
        pass

    def on_train_batch_end(self, batch: int, logs=None):
        pass

    def on_test_begin(self, logs=None):
        pass

    def on_test_end(self, logs=None):
        pass

    def on_test_batch_begin(self, batch: int, logs=None):
        pass

    def on_test_batch_end(self, batch: int, logs=None):
        pass

    def on_epoch_begin(self, epoch: int, logs=None):
        pass

    def on_epoch_end(self, epoch: int, logs=None):
        pass

    def on_batch_begin(self, batch: int, logs=None):
        pass

    def on_batch_end(self, batch: int, logs=None):
        pass

    def get_monitor_value(self, logs: dict):
        logs = logs or {}
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            logging.warning(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s",
                self.monitor,
                ",".join(list(logs.keys())),
            )
        return monitor_value


class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss",min_delta: float = 0.0, 
                 patience: int = 0, verbose: bool = False, mode: str = "min"):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.wait = None
        self.stopped_epoch = None
        self.best = None

        if self.mode not in ["min", "max", "max_equal", "min_equal"]:
            raise ValueError(
                "EarlyStopping mode %s is unknown, "
                "please choose one of min, max, max_equal, min_equal" % self.mode
            )
        
    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf
        self.model.stop_training = False

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.mode == "min" and np.less(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "max" and np.greater(current, self.best + self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "min_equal" and np.less_equal(current, self.best - self.min_delta):
            self.best = current
            self.wait = 0
        elif self.mode == "max_equal" and np.greater_equal(current, self.best + self.min_delta):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            self.logger.info(f"Epoch {self.stopped_epoch}: early stopping")

def assign_mode(mode: str):
    if mode not in ["min", "max", "max_equal", "min_equal"]:
        raise ValueError(
            "ModelCheckpoint mode %s is unknown, "
            "please choose one of min, max, max_equal, min_equal" % mode
        )

    if mode == "min": return np.less
    elif mode == "max": return np.greater
    elif mode == "min_equal": return np.less_equal
    elif mode == "max_equal": return np.greater_equal

class ModelCheckpoint(Callback):
    def __init__(self, filepath: str = None, monitor: str = "val_loss", verbose: bool = False, 
                 save_best_only: bool = True, mode: str = "min") -> None:
                #will save every epoch or only the best

        super(ModelCheckpoint, self).__init__()

        self.filepath = Path(filepath) if filepath else None
        self.monitor = monitor
        self.verbose = verbose
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = None

        self.monitor_op = assign_mode(self.mode)
        
    def on_train_begin(self, logs=None):
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            previous = self.best
            self.best = current
            self.save_model(epoch, current, previous)
        else:
            if not self.save_best_only:
                self.save_model(epoch, current, previous=None)

    def save_model(self, epoch: int, best: float, previous: float = None):

        saved_path = self.model.save(self.filepath)

        if self.verbose:
            if previous is None:
                self.logger.info(f"Epoch {epoch}: {self.monitor} got {best:.5f}, saving model to {saved_path}")
            else:
                self.logger.info(f"Epoch {epoch}: {self.monitor} improved from {previous:.5f} to {best:.5f}, saving model to {saved_path}")

class TensorBoard(Callback):
    def __init__(self, log_dir: str = None, comment: str = None,
                train_name: str = "Train", val_name: str = "Val", train_writer: SummaryWriter = None,
                val_writer: SummaryWriter = None):

        super(TensorBoard, self).__init__()

        self.log_dir = log_dir

        self.train_writer = train_writer
        self.val_writer = val_writer
        self.comment = str(comment) if not None else datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_name = train_name
        self.val_name = val_name

    def on_train_begin(self, logs=None):
        self.log_dir = self.log_dir or self.model.output_path.parent
        if not self.log_dir:
            self.log_dir = "logs"
            self.logging.warning("log_dir not provided. Using default log_dir: logs")

        if self.train_writer is None:
            train_dir = os.path.join(self.log_dir, self.train_name)
            os.makedirs(train_dir, exist_ok=True)
            self.train_writer = SummaryWriter(train_dir, comment=self.comment)

        if self.val_writer is None:
            val_dir = os.path.join(self.log_dir, self.val_name)
            os.makedirs(val_dir, exist_ok=True)
            self.val_writer = SummaryWriter(val_dir, comment=self.comment)

    def update_lr(self, epoch: int):
        for param_group in self.model.optimizer.param_groups:
            if self.train_writer:
                self.train_writer.add_scalar("learning_rate", param_group["lr"], epoch)

    def parse_key(self, key: str):
        if key.startswith("val_"):
            return self.val_name, key[4:]
        else:
            return self.train_name, key

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            if not value:
                continue
            _type, key = self.parse_key(key)
            if _type == self.train_name:
                self.train_writer.add_scalar(key, value, epoch)
            else:
                self.val_writer.add_scalar(key, value, epoch)
        self.update_lr(epoch)

    def on_train_end(self, logs=None):
        self.train_writer.close()
        self.val_writer.close()

class Model2onnx(Callback):


    def __init__(self, input_shape: tuple, saved_model_path: str=None, export_params: bool = True,
                opset_version: int = 14, do_constant_folding: bool = True, input_names: list = ["input"],
                output_names: list = ["output"], dynamic_axes: dict = {"input": {0: "batch_size"},"output": {0: "batch_size"}},
                verbose: bool = False, metadata: dict = None) -> None:
        
        super().__init__()
        self.saved_model_path = saved_model_path
        self.input_shape = input_shape
        self.export_params = export_params
        self.opset_version = opset_version
        self.do_constant_folding = do_constant_folding
        self.input_names = input_names
        self.output_names = output_names
        self.dynamic_axes = dynamic_axes
        self.verbose = verbose
        self.metadata = metadata

    def on_train_end(self, logs=None):
        self.saved_model_path = Path(self.saved_model_path) if self.model else None
        self.onnx_model_path = self.saved_model_path.with_suffix(".onnx") if self.saved_model_path else None

        if not self.saved_model_path.exists():
            self.logger.error(f"Model file not found: {self.saved_model_path}")
            return

        if not self.saved_model_path:
            self.logger.error("Model path not provided. Please provide a path to save the model.")
            return
        
        try:
            # try loading weights from checpoint
            self.model.model.load_state_dict(torch.load(self.saved_model_path))
        except Exception as e:
            self.logger.error(str(e))

        # place model on cpu
        self.model.model.to("cpu")

        # set the model to inference mode
        self.model.model.eval()
        
        # convert the model to ONNX format
        dummy_input = torch.randn((1,) + self.input_shape[1:])

        # handle initial states for LSTM
        lstm_hidden_size = self.model.model.lstm0.hidden_size
        print(lstm_hidden_size)
        h0 = torch.randn(2, 1, lstm_hidden_size)  # 2 for bidirectional
        c0 = torch.randn(2, 1, lstm_hidden_size)  # 2 for bidirectional
        
        dynamic_axes = {
            "input": {0: "batch_size", 1: "sequence_length"},
            "output": {0: "batch_size"},
            "h0": {1: "batch_size"},
            "c0": {1: "batch_size"}
        }

        print(dummy_input.size())
        # Export the model
        torch.onnx.export(
            self.model.model,               
            (dummy_input, h0, c0),                         
            self.onnx_model_path,   
            export_params=self.export_params,        
            opset_version=self.opset_version,          
            do_constant_folding=self.do_constant_folding,  
            input_names = self.input_names + ["h0", "c0"],   
            output_names = self.output_names, 
            dynamic_axes = dynamic_axes,
            )


        if self.verbose:
            self.logger.info(f"Model saved to {self.onnx_model_path}")

        if self.metadata and isinstance(self.metadata, dict):

            onnx_model = onnx.load(self.onnx_model_path)
            # Add the metadata dictionary to the model's metadata_props attribute
            for key, value in self.metadata.items():
                meta = onnx_model.metadata_props.add()
                meta.key = key
                meta.value = str(value)

            # Save the modified ONNX model
            onnx.save(onnx_model, self.onnx_model_path)

        # place model back to original device
        self.model.model.to(self.model.device)

class ReduceLROnPlateau(Callback):
    """ Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates.
    This callback monitors a quantity and if no improvement is seen for a 'patience' number of epochs,
    the learning rate is reduced.
    """
    def __init__(self, monitor: str = "val_loss", factor: float = 0.1, patience: int = 10, min_lr: float = 1e-6, 
                mode: str = "min", verbose: int = False) -> None:
      
        super(ReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.mode = mode

        self.monitor_op = assign_mode(self.mode)

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best = np.inf if self.mode == "min" or self.mode == "min_equal" else -np.Inf

    def on_epoch_end(self, epoch: int, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return
        
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0

        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                current_lr = self.model.optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * self.factor, self.min_lr)
                for group in self.model.optimizer.param_groups:
                    group["lr"] = new_lr
                if self.verbose:
                    self.logger.info(f"Epoch {epoch}: reducing learning rate to {new_lr}.")

class CallbacksHandler:
    #control callback functions during trainng and testing
    def __init__(self, model, callbacks: typing.List[Callback]):
        self.callbacks = callbacks

        # Validate callbacks
        if not all(isinstance(c, Callback) for c in self.callbacks):
            raise TypeError("all items in the callbacks argument must be of type Callback (Check mltu.torch.callbacks.py for more information)")
        
        for callback in self.callbacks:
            callback.model = model
        
    def on_train_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_test_begin(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_begin(logs)

    def on_test_end(self, logs=None):
        for callback in self.callbacks:
            callback.on_test_end(logs)

    def on_batch_begin(self, batch: int, logs=None, train: bool=True):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

            if train:
                callback.on_train_batch_begin(batch, logs)
            else:
                callback.on_test_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs=None, train: bool=True):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

            if train:
                callback.on_train_batch_end(batch, logs)
            else:
                callback.on_test_batch_end(batch, logs)
