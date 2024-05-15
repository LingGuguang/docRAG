
from typing import List
import re

def basic_intent_filter(dirty_intent:str, intent_set: List[str]) -> int:
    if len(dirty_intent) < 5:
        return 0

    for id, intent in enumerate(intent_set):
        if re.findall(str(id), dirty_intent) or re.findall(intent, dirty_intent):
            return id
    
    return 0