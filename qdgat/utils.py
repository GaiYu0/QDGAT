import sys
import random
import torch
import numpy
import logging
from time import gmtime, strftime
from typing import Union, List, Tuple
from drop_eval import (get_metrics as drop_em_and_f1, answer_json_to_strings)

NUM_NER_TYPES = ['ENT2NUM', 'NUMBER', 'PERCENT','MONEY','TIME','DATE','DURATION','ORDINAL', 'YARD']
def dump_gat_info(gnodes_mask, gedges, meta):
   qp_tokens = meta['question_passage_tokens']
   print(meta['original_passage'])
   print('==========mask================')
   gnodes_mask = gnodes_mask.detach().cpu().numpy()
   gnodes = []
   for j,mask in enumerate(gnodes_mask):
     pos = 0
     for i in range(len(mask)):
       if mask[i]>0: pos=i
     gnodes.append((meta['question_passage_tokens'][(mask[0]-1):(mask[pos])], meta['gnodes_type'][j]))
     print(gnodes[-1])
    
   print('==========edges================')
   edges = gedges.detach().cpu().numpy()
   a = {}
   for edge in (gedges.nonzero().detach().cpu().numpy()):
     etype = edges[edge[0], edge[1]]
     src = gnodes[edge[0]]
     dst = gnodes[edge[1]]
     if edge[0] not in a.keys(): 
       a[edge[0]]=[]
     a[edge[0]].append((dst[0],NUM_NER_TYPES[etype-1]))
   for k in a.keys():
     print(gnodes[k],a[k])



def create_logger(name, silent=False, to_disk=True, log_file=None):
    """Logger wrapper
    """
    # setup logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
    if not silent:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        log.addHandler(ch)
    if to_disk:
        log_file = log_file if log_file is not None else strftime("%Y-%m-%d-%H-%M-%S.log", gmtime())
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        log.addHandler(fh)
    return log

def format_number(number):
    if isinstance(number, int):
        return str(number)

    # we leave at most 3 decimal places
    num_str = '%.3f' % number

    for i in range(3):
        if num_str[-1] == '0':
            num_str = num_str[:-1]
        else:
            break

    if num_str[-1] == '.':
        num_str = num_str[:-1]

    # if number < 1, them we will omit the zero digit of the integer part
    if num_str[0] == '0' and len(num_str) > 1:
        num_str = num_str[1:]

    return num_str



def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DropEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """
    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(self, prediction: Union[str, List], ground_truths: List):  # type: ignore
        """
        Parameters
        ----------
        prediction: ``Union[str, List]``
            The predicted answer from the model evaluated. This could be a string, or a list of string
            when multiple spans are predicted as answer.
        ground_truths: ``List``
            All the ground truth answer annotations.
        """
        # If you wanted to split this out by answer type, you could look at [1] here and group by
        # that, instead of only keeping [0].
        ground_truth_answer_strings = [answer_json_to_strings(annotation)[0] for annotation in ground_truths]
        exact_match, f1_score = metric_max_over_ground_truths(
                drop_em_and_f1,
                prediction,
                ground_truth_answer_strings
        )
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official DROP script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __str__(self):
        return f"DropEmAndF1(em={self._total_em}, f1={self._total_f1})"
