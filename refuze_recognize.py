
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
        self.threshold[key] = (hard_threshold, soft_threshold)

    def get_keys(self):
        return self.threshold.keys()
    
    def __getitem__(self, key):
        return self.threshold[key]


class PreNegativeRejection:
    """
        accept soft recognition and hard recognition.
    
    """
    MIN_VALUE = -sys.maxsize-1
    ACCEPT_VS_ACCEPT_AND_SOFT = 0.5

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
        self._check_threshold()

        self.summary_key = summary_key

    def _check_threshold(self):
        [self._check_hard_and_soft(self.threshold[key]) for key in self.threshold.keys()]

    async def _check_hard_and_soft(self, hard_and_soft: Tuple[Any, Any]):
        hard, soft = hard_and_soft
        if hard == None or soft == None:
            return 
        if soft < hard:
            raise ValueError(f'Wrong threshold. You should keep soft > hard, but we got hard {hard} and soft {soft}.')
        

    def run(self, docs_with_scores_set: Dict[str, List[Tuple[str, float]]]) -> Tuple[bool, bool]:
        return asyncio.run(self._run(docs_with_scores_set))

    async def _run(self, docs_with_scores_set: Dict[str, List[Tuple[str, float]]]) -> Tuple[bool, bool]:
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
        tasks = []
        for key in docs_with_scores_set.keys():
            if key not in self.threshold.keys():
                raise ValueError(f"key {key} didn't set threshold.")
            docs_with_scores = docs_with_scores_set[key]
            threshold = self.threshold[key]
            task = asyncio.create_task(self._refuse_tag(rejection_tag, key, docs_with_scores, threshold))
            tasks.append(task)
        await asyncio.gather(*task)

        count = 0
        total_score = 0
        for key in docs_with_scores_set.keys():
            if rejection_tag[key] == 0:
                docs_with_scores_set.pop(key)
            else:
                total_score += rejection_tag[key]
            count += 1
        score = total_score / count
        hard, soft = rejection_tag[self.summary_key]
        if score < hard:
            return True, True 
        elif score > soft:
            return False, True
        else:
            return False, False
        
    async def _refuse_tag(self, refuse_tag, key, docs_with_scores: List[Tuple[str, float]], threshold: Tuple[Optional[float], Optional[float]]) -> List[int]:
        """
            strategy set as described in self.run
        """
        
        hard, soft = threshold
        if not hard and not soft:
            return [2 for _ in len(docs_with_scores)]
        if not hard:
            hard = self.MIN_VALUE
        if not soft:
            soft = hard 
        
        tag = [0, 0]
        for _, score in docs_with_scores:
            if score > soft:
                refuse_tag[key] = 2
                return 
            elif score < hard:
                tag[0] += 1
            else:
                tag[1] += 1 
        if tag[0] < tag[1]:
            refuse_tag[key] = 1
        else:
            refuse_tag[key] = 0
            
