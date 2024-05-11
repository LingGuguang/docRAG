
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from transformers import AutoTokenizer, AutoModel
from typing import cast 

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