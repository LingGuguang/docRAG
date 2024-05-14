from utils.basic_utils import *

## 数据清洗
txt = read_text("史上第一混乱——张小花.txt")
txt = re.split(r"\n\n\n*?", txt)
clean_txt = []
for line in txt:
    line = line.strip()
    if line:
        clean_txt.append(line)

title_re = r"第.季"
chunk_txt = []
temp = ""
for t in clean_txt:
    if re.match(title_re, t):
        chunk_txt.append(temp)
        temp = t + '\n'
    else:
        temp += re.sub("\n", "", t)
chunk_txt.append(temp)

save_txt('史上第一混乱重排版.txt', '\n\n'.join(chunk_txt))