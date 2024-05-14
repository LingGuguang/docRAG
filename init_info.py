
from utils.basic_utils import read_json

class InitInfo:
    intent_file_path = "./utils/intent.json"
    intent_set = [key for key in read_json(intent_file_path).keys()]
    
