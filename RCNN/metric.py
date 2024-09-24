import typing 
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
from itertools import groupby


#accrucacy metrics
def LEDist (pred: typing.List[str], truth: typing.List[str]) -> int:
    #Levenshtein distance Algorithm that measure the number of edits
    #needed to change prediction sentence to match ground truth
    matrix = [[0 for _ in range(len(truth) + 1)] for _ in range(len(pred) + 1)]

    for i in range(len(pred)+1):
        matrix[i][0] = i

    for j in range(len(truth)+1):
        matrix[0][j] = j

    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            
            if p == t:
                matrix[i+1][j+1] = matrix[i][j]
                #number of operations needed is the same with or without this
                #char, so take the operations needed in the prev substring
            else:
                #min number of operations in prev substrings plus operation to this char
                matrix[i+1][j+1] = min(matrix[i][j+1], matrix[i+1][j], matrix[i][j])+1
    return matrix[-1][-1]

def charError(pred: typing.Union[str, typing.List[str]], 
              truth: typing.Union[str, typing.List[str]]) -> float:
    
    if isinstance(pred, str):
        pred = [pred]
    if isinstance(truth, str):
        truth = [truth]

    totalChar, error = 0, 0

    for p_words, t_words in zip(pred, truth):
        error += LEDist(p_words, t_words)
        totalChar += len(t_words)
    
    if totalChar != 0:
        cer = error/totalChar
        return cer
    else:
        return 0

def wordError(pred: typing.Union[str, typing.List[str]], 
              truth: typing.Union[str, typing.List[str]]) -> float:
    
    if isinstance(pred, str) and isinstance(truth, str):
        pred = [pred]
        truth = [truth]

    if isinstance(pred, list) and isinstance(truth, list):
        totalWord, error = 0, 0
        for p_words, t_words in zip(pred, truth):
            if isinstance(p_words, str) and isinstance(t_words, str):
                error += LEDist(p_words.split(), t_words.split())
                totalWord += len(t_words.split())
            else:
                print("Error: preds and target must be either both strings or both lists of strings.")
                return np.inf
    else:
        print("Error: preds and target must be either both strings or both lists of strings.")
        return np.inf
    
    wer = error/totalWord
    return wer


#metric base class
class Metric:
    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self):
        #Reset to initial values and return metric value
        self.__init__()

    def update(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):
        pass

    def result(self):
        pass


class Accuracy(Metric):

    def __init__(self, name="accuracy") -> None:
        super(Accuracy, self).__init__(name=name)
        self.correct = 0
        self.total = 0

    def update(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs):

        _, predicted = torch.max(pred.data, 1)
        self.total += truth.size(0)
        self.correct += (predicted == truth).sum().item()

    def result(self):
        return self.correct / self.total


class CERMetric(Metric):

    def __init__(self, vocabulary: typing.Union[str, list], name: str = "CER") -> None:
        super(CERMetric, self).__init__(name=name)
        self.vocabulary = vocabulary #string of the vocabulary used to encode the labels.
        self.reset()

    def reset(self):
        self.cer = 0
        self.counter = 0

    def update(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs) -> None:
        # convert to numpy
        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()

        # index of the highest probability
        argmax_preds = np.argmax(pred, axis=-1)
        
        # group same indexes
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        # convert indexes to strings
        output_texts = ["".join([self.vocabulary[int(k)] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[int(k)] for k in group if k < len(self.vocabulary)]) for group in truth]

        cer = charError(output_texts, target_texts)
        #print(f"Updating CER: pred={output_texts}, target={target_texts}, cer_value={cer}") #debug

        self.cer += cer
        self.counter += 1

    def result(self) -> float:
        res = self.cer / self.counter
        #print(f"CER Result: {res}") #debug
        return res
    

class WERMetric(Metric):

    def __init__(self, vocabulary: typing.Union[str, list], name: str = "WER") -> None:
        super(WERMetric, self).__init__(name=name)
        self.vocabulary = vocabulary
        self.reset()

    def reset(self):
        self.wer = 0
        self.counter = 0

    def update(self, pred: torch.Tensor, truth: torch.Tensor, **kwargs) -> None:

        pred = pred.detach().cpu().numpy()
        truth = truth.detach().cpu().numpy()

        argmax_preds = np.argmax(pred, axis=-1)
        
        grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

        output_texts = ["".join([self.vocabulary[int(k)] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[int(k)] for k in group if k < len(self.vocabulary)]) for group in truth]

        wer = wordError(output_texts, target_texts)
        #print(f"Updating WER: pred={output_texts}, target={target_texts}, cer_value={wer}")#debug
        self.wer += wer
        self.counter += 1

    def result(self) -> float:
        res = self.wer / self.counter
        #print(f"Wer Result: {res}")#debug
        return res

class MetricsHandler:
    #control metrics during training and testing
    def __init__(self, metrics: typing.List[Metric]):
        self.metrics = metrics

        # Validate metrics
        if not all(isinstance(m, Metric) for m in self.metrics):
            raise TypeError("all items in the metrics argument must be of type Metric (Check mltu.metrics.metrics.py for more information)")
        
        self.train_results_dict = {"loss": None}
        self.train_results_dict.update({metric.name: None for metric in self.metrics})
        
        self.val_results_dict = {"val_loss": None}
        self.val_results_dict.update({"val_" + metric.name: None for metric in self.metrics})

    def update(self, target, output, **kwargs):
        for metric in self.metrics:
            metric.update(output, target, **kwargs)

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def results(self, loss, train: bool=True):
        suffix = "val_" if not train else ""
        results_dict = self.val_results_dict if not train else self.train_results_dict
        results_dict[suffix + "loss"] = loss
        for metric in self.metrics:
            result = metric.result()
            if result:
                if isinstance(result, dict):
                    for k, v in result.items():
                        results_dict[suffix + k] = v
                else:
                    results_dict[suffix + metric.name] = result

        logs = {k: round(v, 4) for k, v in results_dict.items() if v is not None}
        return logs
    
    def description(self, epoch: int=None, train: bool=True):
        epoch_desc = f"Epoch {epoch} - " if epoch is not None else "          "
        dict = self.train_results_dict if train else self.val_results_dict
        return epoch_desc + " - ".join([f"{k}: {v:.4f}" for k, v in dict.items() if v])
    