基于LLM和意图识别的RAG。暂时只能聊天和查chromadb。


意图识别: 
项目初期仅在prompt里用fewshot指导输出+LLM识别意图，准确率讲道理没试过，还行。单字文本会误判，所以加了个过滤器，字符太短就作为闲聊。
随着项目时间变长，收集（query，intention）变多，可以创建query的VD，召回k个相似query，然后做一个LLM的few-shot，以加强意图识别准确度。
当k个召回意图相同时，就可以省去LLM步骤，直接获得意图。但这一切基于有大量（query，意图）数据，且embedding要合适。
当embedding不够满足要求时，要用sft来align query和intention。

[TODO]
拒答模块：
当答案不靠谱时，我们建议模型不回答或软拒答。
不回答，也叫硬拒答，指召回信息(平均或最低)分数低于某个坎时，选择直接省略LLM回复不知道，然后可以选择返回几个分数最高的召回信息，以指导客户重新提问。
软拒答，坎比硬拒答高。举例：0~0.6硬拒答，0.6~0.7软拒答，0.7~1正常回答。软拒答采用prompt的形式拒答，即告诉LLM：query没有answer时，回复不知道。这考验LLM的水平，小模型可能没这个能力还硬答。

如何设置拒答阈值：(手动检查分布，分不开就上微调)
收集应该拒答和不该拒答的query，并与数据库计算相关分数。QQ和QD要分开做。
1.对称召回，即QQ召回，query-question召回，对于FAQ文档很友好。基于倒排索引+bm25可以获得分数。
2.非对称召回，即QD召回，query-document召回，如果拒答/不拒答分数分布差距明显，那么直接看图找阈值。
A.传统召回，ES用的倒排索引+tfidf得分，并用FST(有限状态机)快速查看tfidf所需信息。更好的，用BM25。如何优化？其实要收集负例，看query到底命中了哪些词导致分数失真，然后停用这些词。例如"为什么 的 何时"这些与提问内容没有关系的词。看起来很笨重，但是有时候就是需要一些规则类的方法查漏补缺。
B.embedding召回，用相似度。
如果没法靠人工手调阈值，说明得分函数不对 or embedding不好。此时需要微调。

首先微调embedding，通过双塔模型做align，训练数据如下构建:对称数据（query, 生成的mock_query, neg_query）和非对称数据(instruction+query, 生成的mock_answer, neg_query)，这得基于72B了，否则知识量不够mock_answer生成得不好。除此外，如果有收集用户qa数据，则应划分为训练集和测试集。
然后看看embedding优化之后的分布好不好，能不能拉开分布。如果还不好，说明embedding就卡到这儿了。
然后对reranker微调。
实际上embedding align和reranker是一回事，只不过embedding注重速度，所以是双塔模型。而reranker是cross-encoder，所以性能更好。对他俩的训练本质都是对齐。因此数据和loss都一样。

诚实样本：
最后的generation LLM面对不可靠的上下文可能给出幻觉回答。此时可以通过SFT时加入诚实样本：即上下文与query无关，output诚实地说我不知道的数据样例，以达到拒答目的。


召回策略：

