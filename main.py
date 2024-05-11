# Load model directly
from utils import * 
import torch
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import TextLoader

import chromadb 
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from typing import cast 

import os, sys

### 设定文本
text_name = 'test_data.txt'
text_path = f'dataset/{text_name}'
embedding_model_path = "models/bce-embedding-base_v1"
current_path = os.path.dirname(sys.path[0])
model_path = os.path.join(current_path, "LLModel/baichuan2-7B-chat")
chroma_path = "dataset/"
rerank_model_path = os.path.join(current_path, "LLModel/bce-reranker-base_v1")
RERANK_TOP_K = 3

### 切分chunk
loader = TextLoader(text_path, encoding='utf-8') # 读取Documents格式
docs = loader.load() # 读取Documents格式，服务于split_documents()
docs = read_text(text_path) # 读取string，服务于split_text()

# 一般来说，用RecursiveCharacterTextSplitter把大段的文字粗略地分割成块。这里不要求那么多，只希望章节类文字能大致分开，章节内只需琐碎分开就行。
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=0
)
split_docs = text_splitter.split_text(docs)

# 二次分割，这里会把上一步大块分开的文本细分，此时需要考虑LLM embedding的inuput windows。
text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=16,
    tokens_per_chunk=256,
    model_name=embedding_model_path
)
token_docs = [] 
for doc in split_docs:
    token_docs += text_splitter.split_text(doc)

### 使用chromadb自带的embedding_function

# from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
# embedding_function = SentenceTransformerEmbeddingFunction(device='cuda')

### 自定义本地embedding_function
class bceEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, path : str):
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)

    def __call__(self, docs: Documents):
        input = {k: v for k, v in self.tokenizer(list(docs), padding=True, truncation=True, return_tensors="pt").items()}
        output = self.model(**input, return_dict=True)
        last =  output.last_hidden_state[:, 0]
        return cast(
            Embeddings, 
            last.tolist()
            )
bce_embedding_function = bceEmbeddingFunction(embedding_model_path)

### 可以用client设计pipeline
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(text_name, embedding_function=bce_embedding_function) # 设定文件名(我用文件路径代替)、embedding函数
ids = [str(i) for i in range(len(token_docs))]
chroma_collection.add(ids=ids, documents=token_docs) # 添加ids和切分文件，尺寸要一致
docs_count = chroma_collection.count()

# Retrievel 多路召回,以后接着往里面加,例如BM25之类的
def retrievel(query, n_results=3):
    """
        小科普：Collection里有query和get两个方法。
            query： 可以通过str、embedding、image查找top-n个最相关。
            get：   筛选查找。ids是你给每个文档片段准备的ids，你可以通过ids获取对应的文档片段。每个片段允许有一些元数据(类型、作者之类的)，你可以用它来筛选。
    """
    results = chroma_collection.query(query_texts=query, n_results=n_results, include=['documents', 'embeddings']) # 这里面塞query、embedding，都行，反正embedding_function已经给了
    return results
query = "主角名字是什么？年龄多大？是否有其他名字？"
retrievel_docs = retrievel(query)
print(retrievel_docs['documents'])

# 有时候query与文章并不相似，我们希望通过LLM生成一个伪答案，我们期望这个伪答案与真正的答案长得有一点像，这样就能在向量数据库里找到真正的答案了。
def hypothetical_answer_generation(query: str, model, tokenizer) -> str:
    message = hypothetical_answer_template(query)
    ret = model.chat(tokenizer, message)
    return ret 
# model, tokenizer = init_model(model_path) # get model
# hypothetical_answer = hypothetical_answer_generation(query, model, tokenizer)
# retrievel_docs = retrievel(f'{query} {hypothetical_answer}')
# print(retrievel_docs['documents'])

# 你还可以生成多个表述不同的问题
def additional_query_generation(query: str, model, tokenizer) -> str:
    message = additional_query_template(query)
    ret = model.chat(tokenizer, message)
    return ret
# additional_query = additional_query_generation(query, model, tokenizer)
# retrievel_docs = retrievel(additional_query)
# print(retrievel_docs['documents'])

# 自定义本地rerank model
class bceRerankFunction:
    def __init__(self, path : str):
        # Load model directly
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)

    def __call__(self, docs):
        print(docs)
        input = self.tokenizer(docs, padding=True, truncation=True, return_tensors="pt").items()
        input = {k:v for k,v in input}
        print("——————————————")
        print(input)
        scores = self.model(**input, return_dict=True).logits.view(-1,).float()
        scores = torch.sigmoid(scores)
        print("——————————————")
        print(scores)
        return scores
    
reranker = bceRerankFunction(rerank_model_path)
text_docs = [[query, doc] for doc in retrievel_docs['documents'][0]]
rerank_score = reranker(text_docs)
print("——————rerank score:", rerank_score)
rerank_docs = sorted([(query_and_doc[1], score) for query_and_doc, score in zip(text_docs, rerank_score)], key=lambda x: x[1], reverse=True)
rerank_topk = [doc for doc, score in rerank_docs[:RERANK_TOP_K]]
print("——————rerank_topk:", rerank_topk)

# Augment Generation
def augment_generation(query: str, retrievel_docs: list, model, tokenizer) -> str:
    information = '\n\n'.join(retrievel_docs)
    message = RAG_template_for_baichuan(query, information)
    response = model.chat(tokenizer, message)
    return response 
model, tokenizer = init_model(model_path)
response = augment_generation(query, rerank_topk, model, tokenizer)
print(response)



    










