from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from utils import * 
import re 




def splitter_zh_en(path):
    """
        expire code. for notebook.
    """
    text = read_text(path)
    paper_split_regex = r'\n\n'
    splitted_note = re.split(paper_split_regex, text)

    structed_note = []

    tag_regex = r"\n(?=【.*?】)"
    get_headline_regex = r'(?<=\n【).*?(?=】)'
    replace_headline_regex = r'^【.*?】(\n)?'

    cleaned_notes = []
    for note in splitted_note:
        note = note.strip()
        if not note:
            continue
        # print(note)
        parts = re.split(tag_regex, note)
        part_headline = re.findall(get_headline_regex, note)

        cleaned_note = ""
        if part_headline:
            for part in parts:
                # print(part)
                cleaned_part = re.sub(replace_headline_regex,f'{part_headline.pop(0)}:', part) if re.match(replace_headline_regex, part) else part
                if cleaned_note:
                    cleaned_note += "\n" + cleaned_part
                else:
                    cleaned_note += cleaned_part
        else:
            cleaned_note = note
        # print("\n__________________________________")
        # print(cleaned_note)
        cleaned_notes.append(cleaned_note)
    return "\n\n".join(cleaned_notes)
    

if __name__ == "__main__":
    # path = "dataset.txt"
    # splitted_data = splitter_zh_en(path)
    # save_txt('cleaned_dataset.txt', splitted_data)

    loader = TextLoader('cleaned_dataset.txt', encoding='utf-8')
    docs = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(docs)
    # print(split_docs)