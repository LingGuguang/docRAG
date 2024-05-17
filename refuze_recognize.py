
from typing import Optional, List, Tuple, Union, Any, Dict
import sys 
import asyncio
import numpy as np
from utils.basic_utils import read_json


class Threshold:
    def __init__(self, threshold: Dict[str, Tuple[Optional[float], Optional[float]]] = {}):
        self.threshold = threshold

    def keys(self):
        return [key for key in self.threshold.keys()]

    def add(self, key: str, hard_threshold: Optional[float]=None, soft_threshold: Optional[float]=None):
        temp = {
            "hard" : [],
            "soft" : []
        }
        if hard_threshold:
            temp["hard"] = [hard_threshold]
        if soft_threshold:
            temp['soft'] = [soft_threshold]
        self._check_hard_and_soft(temp)
        self.threshold[key] = temp

    def get_keys(self):
        return self.threshold.keys()
    
    def __getitem__(self, key):
        return self.threshold[key]
    
    def get_threshold_from_key(self, key: str):
        if key in self.threshold.keys():
            hard, soft = self.threshold[key]['hard'], self.threshold[key]['soft']
            if hard == []:
                hard = None 
            else:
                hard = hard[0]
            if soft == []:
                soft == None
            else:
                soft = soft[0]
        else:
            hard, soft = None, None
        return (hard, soft)
    
    def check_threshold(self):
        [self._check_hard_and_soft(self.threshold[key]) for key in self.threshold.keys()]

    def _check_hard_and_soft(self, hard_and_soft: Dict[str, List[float]]):
        hard, soft = hard_and_soft['hard'], hard_and_soft['soft']
        if hard == [] or soft == []:
            return 
        if soft[0] < hard[0]:
            raise ValueError(f'Wrong threshold. You should keep soft > hard, but we got hard {hard[0]} and soft {soft[0]}.')
        


class PreNegativeRejection:
    """
        accept soft recognition and hard recognition.
    
    """
    MIN_VALUE = -sys.maxsize-1
    ACCEPT_VS_ACCEPT_AND_SOFT = 0.5

    REJECT = 0
    SOFT_REJECT = 1
    ACCEPT = 2

    REJECT_RETURN = (True, True)
    SOFT_REJECT_RETURN = (False, False)
    ACCEPT_RETURN = (False, True)

    def __init__(self, 
                 threshold: Threshold = None,
                 threshold_path: str = None,
                 summary_key: str = "summary"):
        if threshold:
            self.threshold = threshold
        elif threshold_path:
            self.threshold = Threshold(read_json(threshold_path))
        else:
            raise ValueError("You must afford threshold or threshold_path.")
        self.threshold.check_threshold()

        self.summary_key = summary_key

    

    def run(self, docs_with_scores_set: Dict[str, List[Tuple[str, float]]]) -> Tuple[bool, bool]:
        """
            策略如下。
            只要某个召回文档组的某个文档过了soft threshold，计2分。
            如果某个召回文档组的文档score卡在hard和soft之间，计1分。
            如果某个文档组的分数都不超过hard threshold，应该剔除该文档组，且计0分。

            为每个文档组打0 1 2分后，计算平均分数，并依靠summary_key记录的threshold判断整体文档的拒答方式。
            计分后，我们一定会删除0分文档组。

            return:
                Tuple[bool, bool]: 
                    (True, Any):      reject answer the question? True.(hard rejection)
                    (False,  False):  must answer the query? False.(soft rejection)
                    (False,  True):   must answer the query? True.(accept)

                    0%->|  (True,Any)  |  (False,False)  |  (False,True)  | <-100%

        """
        rejection_tag = {}
        for key in docs_with_scores_set.keys():
            if key not in self.threshold.keys():
                rejection_tag[key] = self._refuse_tag()
                # raise ValueError(f"key {key} didn't set threshold.")
            scores = [score for _, score in docs_with_scores_set[key]]
            rejection_tag[key] = self._refuse_tag(scores, self.threshold.get_threshold_from_key(key))
        count = 0
        total_score = 0
        for key in docs_with_scores_set.keys():
            if rejection_tag[key] == 0:
                docs_with_scores_set.pop(key)
            else:
                total_score += rejection_tag[key]
            count += 1
        score = total_score / count
        status = self._refuse_tag([score,], self.threshold.get_threshold_from_key(self.summary_key))
        if status == self.REJECT:
            return self.REJECT_RETURN
        elif status == self.ACCEPT:
            return self.ACCEPT_RETURN
        else:
            return self.SOFT_REJECT_RETURN
        
    def _refuse_tag(self, scores: List[float]=None, threshold: Tuple[Optional[float], Optional[float]]=None):
        """
            strategy set as described in self.run
        """
        hard, soft = threshold
        if not hard and not soft:
            return self.ACCEPT
        if not hard:
            hard = self.MIN_VALUE
        if not soft:
            soft = hard 
        
        tag = [0, 0]
        for score in scores:
            print(score, scores, soft)
            if score > soft:
                return self.ACCEPT
            elif score < hard:
                tag[0] += 1
            else:
                tag[1] += 1 
        if tag[0] < tag[1]:
            return self.SOFT_REJECT
        else:
            return self.REJECT
            
