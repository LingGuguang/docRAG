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
print("add embedding into chromadb...") 
chroma_client = chromadb.PersistentClient(chroma_path)
chroma_collection = chroma_client.get_or_create_collection(chromadb_name_for_save, embedding_function=bce_embedding_function) # 设定文件名(我用文件路径代替)、embedding函数

# Retrievel
def retrievel(query, collection ,n_results=3):
    """
        小科普：Collection里有query和get两个方法。
            query： 可以通过str、embedding、image查找top-n个最相关。
            get：   筛选查找。ids是你给每个文档片段准备的ids，你可以通过ids获取对应的文档片段。每个片段允许有一些元数据(类型、作者之类的)，你可以用它来筛选。
    """
    results = collection.query(query_texts=query, n_results=n_results, include=['documents', 'embeddings']) # 这里面塞query、embedding，都行，反正embedding_function已经给了
    return results['documents'][0]
query = "荆轲是什么样的形象？" 
retrievel_docs = retrievel(query, chroma_collection, RETRIEVEL_NUMS)
print("retrievel docs: ", retrievel_docs)

### BM25
def retrievel_BM25(query: str, model: BM25Model):
    ret = model.topk(query)
    return ret
docs = read_text(docs_path, split_line=True)
bm25 = BM25Model(docs)
retrievel_docs += retrievel_BM25(query, bm25)

# 有时候query与文章并不相似，我们希望通过LLM生成一个伪答案，我们期望这个伪答案与真正的答案长得有一点像，这样就能在向量数据库里找到真正的答案了。
def hypothetical_answer_generation(query: str, model, tokenizer) -> str:
    message = hypothetical_answer_template(query)
    ret = model.chat(tokenizer, message)
    return ret 
if parser.hypocritical_answer:
    model, tokenizer = init_model(model_path) # get model
    hypothetical_answer = hypothetical_answer_generation(query, model, tokenizer)
    retrievel_docs += retrievel(f'{query} {hypothetical_answer}', chroma_collection)
    print(retrievel_docs)

# 你还可以生成多个表述不同的问题
def additional_query_generation(query: str, model, tokenizer, query_nums:int=1) -> str:
    message = additional_query_template(query, query_nums)
    ret = model.chat(tokenizer, message)
    return ret
if int(parser.additional_query):
    model, tokenizer = init_model(model_path) # get model
    additional_query = additional_query_generation(query, model, tokenizer, int(parser.additional_query))
    retrievel_docs += retrievel(additional_query, chroma_collection)
    print(retrievel_docs)

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
reranker = bceRerankFunction(rerank_model_path)

text_docs = [[query, doc] for doc in list(set(retrievel_docs))]
rerank_score = reranker(text_docs)
print("——————rerank score:", rerank_score)
rerank_docs = sorted([(query_and_doc[1], score) for query_and_doc, score in zip(text_docs, rerank_score)], key=lambda x: x[1], reverse=True)
rerank_topk_docs = [doc for doc, score in rerank_docs[:RERANK_TOP_K]]
print("——————rerank_topk_docs:", rerank_topk_docs)

# Augment Generation
def augment_generation(query: str, retrievel_docs: list, model, tokenizer) -> str:
    information = '\n\n'.join(retrievel_docs)
    message = RAG_template_for_baichuan(query, information)
    response = model.chat(tokenizer, message)
    return response 
model, tokenizer = init_model(model_path)
response = augment_generation(query, rerank_topk_docs, model, tokenizer)
print(response)





def run(query):









while True:
    query = input("input query: ")
    if query.strip() == "exit":
        break
    request = run(query)
    print(request + '\n')
