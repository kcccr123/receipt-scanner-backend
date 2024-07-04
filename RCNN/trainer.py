import os
import torch
import typing
import traceback
import numpy as np
from qqdm import qqdm, format_str
from pathlib import Path

from metric import MetricsHandler, Metric
from callbacks import Callback, CallbacksHandler
from data import DataLoader

def toTorch(data: np.ndarray, target: np.ndarray) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    #Check if data is of type torch.Tensor, if not convert it to torch.Tensor
    if not isinstance(data, torch.Tensor):
        data = torch.from_numpy(data)

    if not isinstance(target, (torch.Tensor, dict)):
        target = torch.from_numpy(target)

    if data.dtype != torch.float32:
        data = data.float()

    return data, target

class Trainer:
    #Trainer class for training and testing PyTorch neural networks
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, loss: typing.Callable,
                metrics: typing.List[Metric] = [], log_errors: bool = True, output_path: str = None):

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        # get device on which model is running
        self._device = next(self.model.parameters()).device

        self.metrics = MetricsHandler(metrics)

        self.log_errors = log_errors

        self.output_path = Path(output_path) if output_path else None
        if self.output_path:
            if self.output_path.suffix == "":
                self.output_path = Path(os.path.join(self.output_path, "model.pt"))
            os.makedirs(self.output_path.parent, exist_ok=True)

        self.stop_training = False


    def train_step( self, data: typing.Union[np.ndarray, torch.Tensor], 
                    target: typing.Union[np.ndarray, torch.Tensor], loss_info: dict = {}) -> torch.Tensor:

        self.optimizer.zero_grad()#clean up prev step

        output = self.model(data)#forward pass
        loss = self.loss(output, target)#compute loss

        if isinstance(loss, tuple):
            loss, loss_info = loss[0], loss[1:]
        
        loss.backward()#compute parameters

        self.optimizer.step() #update parameters

        if self._device.type == "cuda":
            torch.cuda.synchronize() # synchronize after each forward and backward pass

        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        return loss
    
    def validation_step(self, data: typing.Union[np.ndarray, torch.Tensor], target: typing.Union[np.ndarray, torch.Tensor],
                    loss_info: dict = {} ) -> torch.Tensor:
        output = self.model(data)
        loss = self.loss(output, target)
        if isinstance(loss, tuple):
            loss, loss_info = loss[0], loss[1:]

        self.metrics.update(target, output, model=self.model, loss_info=loss_info)

        # clear GPU memory cache after each validation step
        torch.cuda.empty_cache()

        return loss
    
    def train(self, dataLoader: DataLoader):
        # set model to training mode
        self.model.train()

        loss_sum = 0
        pbar = qqdm(dataLoader, total=len(dataLoader), desc=format_str('bold', f"Epoch {self._epoch}: "))
        for step, (data, target) in enumerate(pbar, start=1):
            self.callbacks.on_batch_begin(step, logs=None, train=True)

            data, target = self.toDevice(*toTorch(data, target))
            loss = self.train_step(data, target)
            loss_sum += loss.item()
            loss_mean = loss_sum / step

            # get training results of one step
            logs = self.metrics.results(loss_mean, train=True)

            # log learning rate into logs
            if len(self.optimizer.param_groups) > 1:
                lr_logs = {f"lr{i}": round(group["lr"], 6) for i, group in enumerate(self.optimizer.param_groups)}
                logs.update(lr_logs)
            else:
                logs["lr"] = round(self.optimizer.param_groups[0]["lr"], 6)

            # update progress bar description
            pbar.set_description(desc=format_str('bold', f"Epoch {self._epoch}: "))
            pbar.set_infos(logs)

            self.callbacks.on_batch_end(step, logs=logs, train=True)

        # reset metrics after each training epoch
        self.metrics.reset()

        # call on_epoch_end of data provider
        dataLoader.on_epoch_end()

        return logs

    def validation(self, dataLoader: DataLoader):
        # set model to evaluation mode
        self.model.eval()
        loss_sum = 0
        pbar = qqdm(dataLoader, total=len(dataLoader), desc=format_str('bold', 'Description'))
        # disable autograd and gradient computation in PyTorch
        with torch.no_grad():
            for step, (data, target) in enumerate(pbar, start=1):
                self.callbacks.on_batch_begin(step, logs=None, train=False)

                data, target = self.toDevice(*toTorch(data, target))
                loss = self.validation_step(data, target)
                loss_sum += loss.item()
                loss_mean = loss_sum / step

                # get testing results of one step
                logs = self.metrics.results(loss_mean, train=False)

                # update progress bar description
                pbar.set_description(f"Epoch {self._epoch}: ")
                pbar.set_infos(logs)

                self.callbacks.on_batch_end(step, logs=logs, train=False)

        # reset metrics after each test epoch
        self.metrics.reset()

        # call on_epoch_end of data provider
        dataLoader.on_epoch_end()

        return logs
    
    def save(self, path: str=None):

        if not path and not self.output_path:
            print("Path to file is not provided, model will not be saved") # replace to error logging
            return
        
        model_to_save = self.model
        
        output_path = Path(path or self.output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        model_to_save.eval()
        try:
            torch.save(model_to_save.state_dict(), output_path)
            return str(output_path)
        except Exception:
            traceback.print_exc()
            torch.save(model_to_save, output_path.with_suffix(".pth"))
            return str(output_path.with_suffix(".pth"))
    
    def run(self, train_dataLoader: DataLoader, validation_dataLoader: DataLoader=None, 
        epochs: int=10, initial_epoch: int = 1, callbacks: typing.List[Callback] = []) -> dict:

        self._epoch = initial_epoch
        history = {}
        self.callbacks = CallbacksHandler(self, callbacks)
        self.callbacks.on_train_begin()
        for epoch in range(initial_epoch, initial_epoch + epochs):
            self.callbacks.on_epoch_begin(epoch)

            train_logs = self.train(train_dataLoader)
            val_logs = self.validation(validation_dataLoader) if validation_dataLoader else {}

            logs = {**train_logs, **val_logs}
            self.callbacks.on_epoch_end(epoch, logs=logs)

            if self.stop_training:
                break

            history[epoch] = logs
            self._epoch += 1

        self.callbacks.on_train_end(logs)

        return history
    
    def validate(self, dataLoader: DataLoader, initial_epoch: int = 1,
        callbacks: typing.List[Callback] = []) -> dict:

        self._epoch = initial_epoch
        self.callbacks = CallbacksHandler(self, callbacks)
        self.model.eval()
        logs = self.validation(dataLoader)
        return logs