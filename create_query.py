import chromadb 
from llm import bceEmbeddingFunction, myChain, QwenLLMChat
import os, sys
from utils.get_prompt import generate_query_from_dataset

chromadb_name_for_save = "chaos_history"
chroma_path = "dataset/"
embedding_model_path = "models/bce-embedding-base_v1"
print("get chromadb...") 
current_path = os.path.dirname(sys.path[0])
model_name = "qwen1.5-14B-chat"



model_path = os.path.join(current_path, f"LLModel/{model_name}")
chroma_client = chromadb.PersistentClient(chroma_path)
bce_embedding_function = bceEmbeddingFunction(embedding_model_path)
chroma_collection = chroma_client.get_or_create_collection(name=chromadb_name_for_save, embedding_function=bce_embedding_function)
# llm = QwenLLMChat(model_path)
# QAgeneraterChain = myChain(llm=llm,
#                 prompt=generate_query_from_dataset(),
#                 # memory=self.memory
#                 )

datas = chroma_collection.get()
print(datas[0])


# response = QAgeneraterChain.invoke()






    