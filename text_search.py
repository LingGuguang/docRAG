from rank_bm25 import BM25Okapi
import jieba
from typing import List, Optional, Tuple
import torch

class BM25Model:
    def __init__(self, data_list:List[List[str]]):
        self.data_list = data_list
        # corpus : list of list of str
        self.corpus = self._load_corpus()
        self.bm = BM25Okapi(self.corpus)

    def topk(self, query, k=1) -> Tuple[List[str], List[float]]:
        """
            return (List[docs], List[scores])
        
        """
        query = jieba.lcut(query)  # 分词
        scores = self.bm.get_scores(query)
        topk_scores, topk_ids = torch.topk(torch.Tensor(scores), k=k)
        
        return ([self.data_list[id] for id in topk_ids], 
                [score.item() for score in topk_scores])


    def _load_corpus(self) -> List[List[str]]:
        corpus = [jieba.lcut(data) for data in self.data_list]
        return corpus


if __name__ == '__main__':
    data_list = ["小丁的文章不好看", "我讨厌小丁的创作内容", "我非常喜欢小丁写的文章"]
    BM25 = BM25Model(data_list)
    query = "小丁的文章不好看"
    print(BM25.bm25_similarity(query, 1))