
from utils.basic_utils import read_json
from utils.get_memory import SUI_MEMORY

class InitInfo:
    intent_file_path = "./utils/intent.json"
    intent_set = read_json(intent_file_path).keys()
    

class promptInfo:
    rag_text = None 
    intent = None 
    def __call__(self):
        return self.__dict__