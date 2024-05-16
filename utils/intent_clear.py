
from typing import List
import re

SHORT_QUERY_AS_CHAT = 5

def basic_query_intention_filter(init_query: str):
    if len(list(init_query)) < SHORT_QUERY_AS_CHAT:
        


def intent_chain_after_filter(dirty_intent:str, intent_set: List[str]) -> int:
    for id, intent in enumerate(intent_set):
        if re.findall(str(id), dirty_intent) or re.findall(intent, dirty_intent):
            return id
    return 0