# ADL 147
## Report 1. 知识检索增强：范式与关键技术

同济大学 王昊奋

Retrieval-Augmented Generation for Large Language Models: A Survey  [arXiv24.3.27](https://arxiv.org/abs/2312.10997)

### 01 RAG基本概述

#### RAG提出背景

LLM缺陷：幻觉（一本正经的胡说八道）、信息过时、参数化知识效率低、缺乏专业领域的深度知识、推理能力弱

实际应用的需求：领域精准问答、数据频繁更新、生成内容可解释可溯源、成本可控、数据隐私保护

![image-20240525180831886](imgs/image-20240525180831886.png)

### 02 RAG的主要范式与发展历程

![image-20240525181145792](imgs/image-20240525181145792.png)

#### RAG的三种典型范式：

- Naive RAG：

  - 构建数据索引：文档分块、生成embedding、存储到向量数据库中
  - 检索：向量相似度度量，得到k个文档
  - 原始的query与检索得到的文本组合输入到LLM，得到回答

- Advanced RAG

  - 索引优化：滑动窗口、细粒度分割、元数据
  - 前检索模块：检索路由、摘要、问题重写、置信度判断
  - 后检索模块：重排序、检索内容过滤

- 模块化RAG

  ![image-20240525181707492](imgs/image-20240525181707492.png)

#### RAG三大问题：

- 检索什么？词元、词组、句子、段落、实体、知识图谱

  ![image-20240525182013471](imgs/image-20240525182013471.png)

- 什么时候检索？单次检索、每个token、每N个token、自适应检索

  ![image-20240525182343538](imgs/image-20240525182343538.png)

- 怎么使用检索的结果？输入/数据层、模型/中间层、输出/预测层

![image-20240525182406510](imgs/image-20240525182406510.png)

### 03 模块化RAG范式与关键技术

![image-20240525182511261](imgs/image-20240525182511261.png)

#### 1. Indexing 索引

面临挑战：文档块不完整的语义信息、块相似度计算不准确、参考轨迹不明确

解决方案：分块索引优化、结构化语料

**分块索引优化：**

![image-20240525182857082](imgs/image-20240525182857082.png)

**结构化语料：**

![image-20240525182911100](imgs/image-20240525182911100.png)

#### 2. Pre-Retrieval 检索前处理

面临挑战：措辞不当的查询、语言自身的复杂性和歧义性

解决方案：查询扩展、查询转换、查询路由

**查询转换：**

![image-20240525183331007](imgs/image-20240525183331007.png)

**查询拓展：**

![image-20240525183350124](imgs/image-20240525183350124.png)

**查询路由：**

![image-20240525183425789](imgs/image-20240525183425789.png)

#### 3. Retrieval 检索

面临挑战：检索效率、嵌入表示质量、任务 数据和模型的一致性

解决方案：检索元选择、检索器选择、检索微调

**检索元选择：**![image-20240525183750337](imgs/image-20240525183750337.png)

**检索器选择：**

![image-20240525183829989](imgs/image-20240525183829989.png)

**检索器微调：**

![image-20240525183858160](imgs/image-20240525183858160.png)

#### 4. Post-Retrieval 后检索

面临挑战：噪音/反事实文档、上下文窗口影响

解决方案：重排序rerank、上下文压缩筛选、检索器微调

重排序：

![image-20240525192227690](C:/Users/bryce/AppData/Roaming/Typora/typora-user-images/image-20240525192227690.png)

压缩与筛选：

![image-20240525192242045](C:/Users/bryce/AppData/Roaming/Typora/typora-user-images/image-20240525192242045.png)

![image-20240525192252425](imgs/image-20240525192252425.png)

#### 5. Generation 生成

面临挑战：LLM的选型、缺乏领域知识、复杂问题推理能力有限、LLM的幻觉

解决方案：生成器选型、生成器微调、事实校验与知识编辑

模型选型：

![image-20240525192456565](C:/Users/bryce/AppData/Roaming/Typora/typora-user-images/image-20240525192456565.png)

生成器微调：

![image-20240525192511748](imgs/image-20240525192511748.png)

事实校验：

![image-20240525192530447](imgs/image-20240525192530447.png)

#### 6. Orchestration 编排

面临挑战：传统的链式且一次性的检索-生成流程不足以解决复杂推理或设计大量知识的任务

解决方案：检索流程调度、知识引导检索流程、检索流程聚合

检索流程调度：

![image-20240525192740195](imgs/image-20240525192740195.png)

知识引导：

![image-20240525192801820](imgs/image-20240525192801820.png)

流程聚合：

![image-20240525192822719](imgs/image-20240525192822719.png)

#### 7. RAG FLOW

![image-20240525193309069](imgs/image-20240525193309069.png)

##### 线性RAG FLOW

![image-20240525193136443](imgs/image-20240525193136443.png)

##### 条件RAG FLOW

![image-20240525193202789](imgs/image-20240525193202789.png)

##### 分支RAG FLOW

![image-20240525193225170](imgs/image-20240525193225170.png)

##### 环状RAG FLOW

![image-20240525193244170](imgs/image-20240525193244170.png)



### 04 RAG的下游任务与评估

#### 常用数据集：

![image-20240525193359704](imgs/image-20240525193359704.png)

#### 下游任务：

![image-20240525193414555](imgs/image-20240525193414555.png)

#### 评估方法:

![image-20240525193447160](imgs/image-20240525193447160.png)

目前评估依赖于WiKipedia Dump数据集（40G），进行评估成本较高，并且很多评估数据集的语料库都用于模型训练存在泄露问题；外面需要更低成本的更公正的面向RAG的评测体系。

### 05 RAG的工具栈与行业应用

#### 个人知识助手

Quivr: Your second brain

WhyHowAI

#### 问答系统 Linkdin智能客服

#### 软件工程 CodeGeeX

#### 技术栈与工具

![image-20240525194013821](imgs/image-20240525194013821.png)

### 06 RAG的挑战与机遇

#### 理论侧：

外部知识和内化知识的博弈、RAG记忆与遗忘机制

![image-20240525194203843](imgs/image-20240525194203843.png)

#### 数据侧：

多模态数据检索与理解、长文本的高效切分 向量化与检索、大规模知识库的管理、异构数据的整合与存储、评测集与指标

![image-20240525194237089](imgs/image-20240525194237089.png)

#### 技术侧：

幻觉控制、隐私保护、自主控制的RAG流程、流程可追溯 结果可解释、查询与知识的语义距离

![image-20240525194543233](imgs/image-20240525194543233.png)

![image-20240525194558372](imgs/image-20240525194558372.png)

#### 应用侧：

大规模工业应用、低资源任务场景

![image-20240525194608159](imgs/image-20240525194608159.png)



## Report 2. 大语言模型与智能信息检索技术的融合

中国人民大学高瓴人工智能学院 窦志成  朱余韬

### PART 1 大模型赋能的信息检索

Large Language Models for Information Retrieval: A Survey  [arXiv23.8.14](https://arxiv.org/abs/2308.07107)

![image-20240525221211996](imgs/image-20240525221211996.png)

#### 1. Rewriter  查询改写

为什么要进行查询改写？

- 原查询过短或模糊，大模型可以更好的理解用户意图
- 在对话系统中，改写更为重要，继承上下文

![image-20240525222641445](imgs/image-20240525222641445.png)

基于LLM进行对话式查询改写

>  Large Language Models Know Your Contextual Search Intent:A Prompting Framework for Conversational Search [EMNLP 2023](https://arxiv.org/abs/2303.06573)
>
> ![image-20240525223749517](imgs/image-20240525223749517.png)

![image-20240525222934341](imgs/image-20240525222934341.png)



#### 2. Retriever 检索器

从海量文档中**高效高质**的返回相关结果.

挑战：查询模糊、文档内容多、信息复杂、标注开销大等

##### 基于大模型生成检索器的训练数据

![image-20240525224647271](imgs/image-20240525224647271.png)

##### 以大模型为基座的检索模型

Task-aware Retrieval with Instructions [arXiv2211](https://arxiv.org/abs/2211.09260)

![image-20240525224843266](imgs/image-20240525224843266.png)

##### 大模型改善生成式检索

Transformer Memory as a Differentiable Search Index [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/892840a6123b5ec99ebaab8be1530fba-Abstract-Conference.html)

![image-20240525225119959](imgs/image-20240525225119959.png)

Large Language Models are Built-in Autoregressive Search Engines [ACL 2023](https://arxiv.org/abs/2305.09612)

![image-20240525225312847](imgs/image-20240525225312847.png)

#### 3. Reranker 重排序器

##### 微调大模型做重排序  

通常有好的性能，但训练开销大

![image-20240525225432840](imgs/image-20240525225432840.png)

##### 提示大模型做重排序

需要大模型能力足够强大

![image-20240525225551077](imgs/image-20240525225551077.png)

##### 大模型生成排序数据

提升已有排序模型的有效策略

ExaRanker: Explanation-Augmented Neural Ranker  [arXiv2301](https://arxiv.org/abs/2301.10521)

使用chatGPT生成解释作为额外的标签

![image-20240525225726277](imgs/image-20240525225726277.png)

#### 4. Reader 阅读器

基于大模型对检索到的文档进行提炼总结，得到最终的答案输出

- 新型搜索引擎
  - New bing  百度AI搜索
- 商业大模型
  - Kimi chat   Baichuan
- 效果仍有巨大提升空间
  - 幻想
  - 引用不相关内容
  - 编造内容
  - 错误编号

#### 5. 搜索Agent

进入强化学习的思想

##### 静态Agent

将人类浏览网页的过程拆解为子过程逐模块的使用Agent进行模拟

![image-20240525230315520](imgs/image-20240525230315520.png)

##### 动态Agent

智能体自行决定行为

![image-20240525230423091](imgs/image-20240525230423091.png)

#### 6. ACL24 面向信息检索任务的指令微调

INTERS: Unlocking the Power of Large Language Models in Search with Instruction Tuning

![image-20240525230912186](imgs/image-20240525230912186.png)

![image-20240525230925628](imgs/image-20240525230925628.png)

### Part 2 检索增强的生成大模型

#### 为什么要做检索增强？

- 大模型并不完美  幻觉问题  知识缺陷  时效性问题

未应用检索增强的大模型（左）笼统的套话+乱说，应用检索增强的大模型（右图）能根据查询到的文档来给出问题的答案

![image-20240525231319755](imgs/image-20240525231319755.png)

#### RAG的基本框架

![image-20240525231547537](imgs/image-20240525231547537.png)

#### 何时需要检索？ --检索的必要性判定

##### ACL24 SlimPLM 代理模型判定检索的必要性

Small Models, Big Insights: Leveraging Slim Proxy Models To Decide When and What to Retrieve for LLMs  2024 ACL

- 检索一定能增强大模型的生成效果么？
  - 无关结果会带来负面的影响
  - 大模型能够掌握的知识不需要检索

- 先前方法：先生成判定结果后在检索生成，这种方式成本高

- 实验发现在模型置信度较高时无论模型规模大小模型预测结果相似

  ![image-20240525232235679](imgs/image-20240525232235679.png)

- 提出使用较小的语言模型作为代理模型，根据代理模型的表现来判定需要做检索的时机

![image-20240525232322966](imgs/image-20240525232322966.png)

#### 检索结果如何使用？ --精炼结果的方法

##### ACL24 BIDER 为大模型精炼有效知识

BIDER: Bridging Knowledge Inconsistency for Efficient Retrieval-Augmented LLMs via Key Supporting Evidence  ACL 2024

- 造成RAG下降的原因：大语言模型与检索系统之间的知识不一致
  - 检索文档往往冗长且含有噪声
  - LLM无法感知自身的知识边界
- 现有方法过度依赖外部知识（检索）或内部知识（LLM）的一部分而忽略了两者之间的联系
  - 基于困惑度的方法：保留LLM认为有高困惑度的词语
  - 基于模型的方法：保留检索系统判断与正确答案最接近的句子
- 构建模型对检索文档进行精炼，提供LLM最需要的知识

![image-20240525233105184](imgs/image-20240525233105184.png)

#### 较长的结果如何建模？ --无切块长文本建模方法

##### ACL24 CFIC 建模更长的检索结果

Grounding Language Model with Chunking-Free In-Context Retrieval  ACL 2024

- 检索结果通常比较长，传统RAG处理长上下文时存在局限性
  - LLM无法处理超长文本
  - 长上下文中存在大量的不相关内容
- 已有方法
  - 提高上下文窗口大小直接处理
  - 将长上下文切块和排序，寻找相关内容
- 问题
  - 直接处理长上下文算力要求高，且无法消除长上下文中的噪音
  - 切块破坏了语义的连贯性，导致信息缺失
- 提出一种高效的上下文提炼的方法，不破坏语意连贯性且能高效找到支持回复生成的文本证据

![image-20240525234026079](imgs/image-20240525234026079.png)

#### 工具包

##### arXiv24.5.24 FlashRAG 快速实现RAG方法工具包

FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research  

动机：

- RAG系统组件众多，研究人员要花费大量时间在各类工程的实现上
- 现有的RAG工作缺少统一的实现框架，导致复现非常耗时且难以公平比较
- 已有的LangChain LlambIndex工具封装复杂，难以满足定制化研究需求

特点：

- 模块化RAG框架，包含检索器、生成器、精炼器等多种组件，支持自定义RAG流程
- 目前实现12种RAG工作，轻松在不同设置下评估结果
- 包含32个RAG工作中常用数据集，并预处理为统一的格式
- 包含多种辅助脚本，包含Wikipedia预处理分块、索引构建、检索结果预准备等

![image-20240525234840677](imgs/image-20240525234840677.png)

![image-20240525234853516](imgs/image-20240525234853516.png)

#### 数据集

#### WebBrain 面向RAG的通用数据集

WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus arXiv2304

- 现有的RAG数据集，尤其是训练集不足

  - 已有工作多采用open-domain QA 作为训练和测试集
  - 人为使用检索器构建训练集合，检索和生成文本的关联性缺乏保障
  - 难以判断是否参考了检索结果

- 提出基于维基百科的文本及其引用构建大规模数据集

  - 包括对维基百科引用链接进行标注

  ![image-20240525235636953](imgs/image-20240525235636953.png)

#### DomainRAG 特定领域RAG评测

DomainRAG: A Chinese Benchmark for Evaluating Domain-specific Retrieval-Augmented Generation  ACL 2024、

动机：

- RAG能够有效解决LLM的各种限制，例如幻觉和知识实时更新的困难
- 目前的研究往往依赖于维基百科等一般知识源来评估模型解决常识性问题的能力，然而RAG在LLM难以涵盖专业知识的场景和特定领域中的应用也很重要

方法：

- 使用特定领域的语料库和问题对于评估LLM有效利用来自这些特定领域的外部知识来解决专家问题的能力至关重要
- 总结综合评价RAG模型的六个重要能力，并以人大招生为应用场景构建了评估这些能力的数据集
- ![image-20240526000350164](imgs/image-20240526000350164.png)

### Future Work

- 更加精准的查询分解与改写
- 对话式RAG的进一步探索
- 面向RAG的训练 （预训练、指令微调）
- 长窗口与RAG之间的关联
- RAG系统的评估fang'fa

### Part 3 生成式文档检索

#### Classical Work：DSI (Google)
#### Classical Work: WebUltron (renda)
#### Classical Work: NCI (MicroSoft)

#### 跳出Sequence范式，词袋模型

#### 可学习的文档标识符

#### CorpusLM
