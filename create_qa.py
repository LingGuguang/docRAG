import chromadb 
from llm import bceEmbeddingFunction, myChain, QwenLLMChat
import os, sys
from utils.basic_utils import save_json, save_text
from utils.get_prompt import generate_query_from_dataset

chromadb_name_for_save = "chaos_history"
chroma_path = "dataset/"
embedding_model_path = "models/bce-embedding-base_v1"
print("get chromadb...") 
current_path = os.path.dirname(sys.path[0])
model_name = "qwen1.5-14B-chat"

save_qa = 'dataset/generate_qa.json'



model_path = os.path.join(current_path, f"LLModel/{model_name}")
chroma_client = chromadb.PersistentClient(chroma_path)
bce_embedding_function = bceEmbeddingFunction(embedding_model_path)
chroma_collection = chroma_client.get_or_create_collection(name=chromadb_name_for_save, embedding_function=bce_embedding_function)
llm = QwenLLMChat(model_path)
QAgeneraterChain = myChain(llm=llm,
                prompt=generate_query_from_dataset(),
                # memory=self.memory
                )

data = chroma_collection.get()
data = [(ids, docs) for ids, docs in zip(data['ids'], data['documents'])] 
print(len(data))
print(data[0])

generate_qa_list = []
for ids, docs in data[:10]:
    temp = {}
    response = QAgeneraterChain.invoke(docs)
    generate_qa_list.append(response)






    