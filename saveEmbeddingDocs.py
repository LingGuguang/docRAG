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
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from llm import bceEmbeddingFunction

### 设定文本
text_name = '史上第一混乱重排版.txt'
chromadb_name_for_save = "chaos_history"
text_path = f'dataset/{text_name}'
embedding_model_path = "models/bce-embedding-base_v1"
chroma_path = "dataset/"

### 切分chunk
loader = TextLoader(text_path, encoding='utf-8') # 读取Documents格式
docs = loader.load() # 读取Documents格式，服务于split_documents()
docs = read_text(text_path) # 读取string，服务于split_text()

# 一般来说，用RecursiveCharacterTextSplitter把大段的文字粗略地分割成块。这里不要求那么多，只希望章节类文字能大致分开，章节内只需琐碎分开就行。
print("split doc 1/2...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=0
)
split_docs = text_splitter.split_text(docs)

# 二次分割，这里会把上一步大块分开的文本细分，此时需要考虑LLM embedding的inuput windows。
print("split doc 2/2...")
text_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=16,
    tokens_per_chunk=256,
    model_name=embedding_model_path
)
token_docs = [] 
for doc in split_docs:
    token_docs += text_splitter.split_text(doc)

### 使用chromadb自带的embedding_function
### 很垃圾，别用，很多token根本不识别
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction(device='cuda')

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
print("bce embedding is running...")    
bce_embedding_function = bceEmbeddingFunction(embedding_model_path)

### 可以用client设计pipeline
print("add embedding into chromadb...") 
chroma_client = chromadb.PersistentClient(chroma_path)
# chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(chromadb_name_for_save, embedding_function=bce_embedding_function) # 设定文件名(我用文件路径代替)、embedding函数
ids = [str(i) for i in range(len(token_docs))]
print(len(ids))
print("adding...")
for ids in tqdm(range(len(token_docs)),desc='进度',unit='单位'):
    chroma_collection.add(ids=[str(ids),], documents=[token_docs[ids],]) # 添加ids和切分文件，尺寸要一致
docs_count = chroma_collection.count()
print(f"total nums of embedding in {chromadb_name_for_save} chroma:", docs_count)




    










