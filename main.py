# Load model directly
from utils import * 
import torch
from transformers import  AutoTokenizer, AutoModelForSequenceClassification
import chromadb 
import os, sys
from llm import bceEmbeddingFunction

from argparser import main_argparser
from text_search import BM25Model

os.environ["TOKENIZERS_PARALLELISM"] = "false"





# 自定义本地rerank model
class bceRerankFunction:
    def __init__(self, path : str):
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

    def __call__(self, docs):
        input = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt").items()
        input = {k:v for k,v in input}
        scores = self.model(**input, return_dict=True).logits.view(-1,).float()
        scores = torch.sigmoid(scores)
        return scores


class docRAG:
    parser = main_argparser()

    ### 设定文本
    chromadb_name_for_save = "chaos_history"
    dataset_dirname = 'dataset'
    docs_path = f'{dataset_dirname}/史上第一混乱碎片版.txt'

    current_path = os.path.dirname(sys.path[0])
    embedding_model_path = "models/bce-embedding-base_v1"
    model_path = os.path.join(current_path, "LLModel/baichuan2-7B-chat")
    rerank_model_path = os.path.join(current_path, "LLModel/bce-reranker-base_v1")
    chroma_path = "dataset/"

    RERANK_TOP_K = 3
    RETRIEVEL_NUMS=3

    ### 自定义本地embedding_function
    print("bce embedding is running...")    
    bce_embedding_function = bceEmbeddingFunction(embedding_model_path)

    ### 可以用client设计pipeline
    print("get chromadb...") 
    chroma_client = chromadb.PersistentClient(chroma_path)
    chroma_collection = chroma_client.get_or_create_collection(chromadb_name_for_save, embedding_function=bce_embedding_function) # 设定文件名(我用文件路径代替)、embedding函数

    docs = read_text(docs_path, split_line=True)
    reranker = bceRerankFunction(rerank_model_path)

    model, tokenizer = None, None # get model

    def run(self, query: str) -> str: 
        retrievel_docs = self.retrievel(query, n_results=self.RETRIEVEL_NUMS)
        # print("retrievel docs: ", retrievel_docs)

        bm25 = BM25Model(self.docs)
        retrievel_docs += self.retrievel_BM25(query, bm25)
        if self.parser.hypocritical_answer:
            # if not self.model or not self.tokenizer:
            #     self.model, self.tokenizer = init_model(self.model_path) # get model
            hypothetical_answer = self.hypothetical_answer_generation(query)
            retrievel_docs += self.retrievel(f'{query} {hypothetical_answer}')
            # print(retrievel_docs)
        if int(self.parser.additional_query):
            # if not self.model or not self.tokenizer:
            #     model, tokenizer = init_model(self.model_path) # get model
            additional_query = self.additional_query_generation(query, int(self.parser.additional_query))
            retrievel_docs += self.retrievel(additional_query)
            # print(retrievel_docs)

        text_docs = [[query, doc] for doc in list(set(retrievel_docs))]
        rerank_score = self.reranker(text_docs)
        # print("——————rerank score:", rerank_score)
        rerank_docs = sorted([(query_and_doc[1], score) for query_and_doc, score in zip(text_docs, rerank_score)], key=lambda x: x[1], reverse=True)
        rerank_topk_docs = [doc for doc, score in rerank_docs[:self.RERANK_TOP_K]]
        # print("——————rerank_topk_docs:", rerank_topk_docs)

        # if not self.model or not self.tokenizer:
        #     self.model, self.tokenizer = init_model(self.model_path)
        response = self.augment_generation(query, rerank_topk_docs, model, tokenizer)

        return response
    
    # Retrievel
    def retrievel(self, query: str, n_results: int=3) -> List[str]:
        """
            小科普：Collection里有query和get两个方法。
                query： 可以通过str、embedding、image查找top-n个最相关。
                get：   筛选查找。ids是你给每个文档片段准备的ids，你可以通过ids获取对应的文档片段。每个片段允许有一些元数据(类型、作者之类的)，你可以用它来筛选。
        """
        results = self.chroma_collection.query(query_texts=query, n_results=n_results, include=['documents', 'embeddings']) # 这里面塞query、embedding，都行，反正embedding_function已经给了
        return results['documents'][0]

    ### BM25
    def retrievel_BM25(self, query: str, model: BM25Model, n_results: int=3) -> List[str]:
        ret = model.topk(query, k=n_results)
        return ret
    
    def model_check(func):
        def wrapper(self, *args, **kwargs):
            if not self.model or not self.tokenizer:
                self.model, self.tokenizer = init_model(self.model_path)
            return func(self, *args, **kwargs)
        return wrapper

    # 有时候query与文章并不相似，我们希望通过LLM生成一个伪答案，我们期望这个伪答案与真正的答案长得有一点像，这样就能在向量数据库里找到真正的答案了。
    @model_check
    def hypothetical_answer_generation(self, query: str) -> str:
        message = hypothetical_answer_template(query)
        ret = self.model.chat(self.tokenizer, message)
        return ret 


    # 你还可以生成多个表述不同的问题
    @model_check
    def additional_query_generation(self, query: str, query_nums:int=1) -> str:
        message = additional_query_template(query, query_nums)
        ret = self.model.chat(self.tokenizer, message)
        return ret
    
    # Augment Generation
    @model_check
    def augment_generation(self, query: str, retrievel_docs: list) -> str:
        information = '\n\n'.join(retrievel_docs)
        message = RAG_template_for_baichuan(query, information)
        response = self.model.chat(self.tokenizer, message)
        return response 


rag = docRAG()

while True:
    query = input("input query: ")
    if query.strip() == "exit":
        break
    request = rag.run(query)
    print(request + '\n')
