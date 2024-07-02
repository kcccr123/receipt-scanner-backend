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
                matrix[i+1][j+1] = min(matrix[i][j+1], matrix[i+1][j+1], matrix[i][j])+1
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
        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in truth]

        cer = charError(output_texts, target_texts)

        self.cer += cer
        self.counter += 1

    def result(self) -> float:
        return self.cer / self.counter
    

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

        output_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in grouped_preds]
        target_texts = ["".join([self.vocabulary[k] for k in group if k < len(self.vocabulary)]) for group in truth]

        wer = wordError(output_texts, target_texts)

        self.wer += wer
        self.counter += 1

    def result(self) -> float:
        return self.wer / self.counter