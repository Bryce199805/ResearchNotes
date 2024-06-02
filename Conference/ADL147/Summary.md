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

#### 1. 为什么要做检索增强？

- 大模型并不完美  幻觉问题  知识缺陷  时效性问题

未应用检索增强的大模型（左）笼统的套话+乱说，应用检索增强的大模型（右图）能根据查询到的文档来给出问题的答案

![image-20240525231319755](imgs/image-20240525231319755.png)

#### 2. RAG的基本框架

![image-20240525231547537](imgs/image-20240525231547537.png)

#### 3. 何时需要检索？ --检索的必要性判定

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

#### 4. 检索结果如何使用？ --精炼结果的方法

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

#### 5. 较长的结果如何建模？ --无切块长文本建模方法

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

#### 6. 工具包

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

#### 7. 数据集

##### WebBrain 面向RAG的通用数据集

WebBrain: Learning to Generate Factually Correct Articles for Queries by Grounding on Large Web Corpus arXiv2304

- 现有的RAG数据集，尤其是训练集不足

  - 已有工作多采用open-domain QA 作为训练和测试集
  - 人为使用检索器构建训练集合，检索和生成文本的关联性缺乏保障
  - 难以判断是否参考了检索结果

- 提出基于维基百科的文本及其引用构建大规模数据集

  - 包括对维基百科引用链接进行标注

  ![image-20240525235636953](imgs/image-20240525235636953.png)

##### DomainRAG 特定领域RAG评测

DomainRAG: A Chinese Benchmark for Evaluating Domain-specific Retrieval-Augmented Generation  ACL 2024、

动机：

- RAG能够有效解决LLM的各种限制，例如幻觉和知识实时更新的困难
- 目前的研究往往依赖于维基百科等一般知识源来评估模型解决常识性问题的能力，然而RAG在LLM难以涵盖专业知识的场景和特定领域中的应用也很重要

方法：

- 使用特定领域的语料库和问题对于评估LLM有效利用来自这些特定领域的外部知识来解决专家问题的能力至关重要

- 总结综合评价RAG模型的六个重要能力，并以人大招生为应用场景构建了评估这些能力的数据集

  ![image-20240526000350164](imgs/image-20240526000350164.png)

#### 8. Future Work

- 更加精准的查询分解与改写
- 对话式RAG的进一步探索
- 面向RAG的训练 （预训练、指令微调）
- 长窗口与RAG之间的关联
- RAG系统的评估fang'fa

### Part 3 生成式文档检索

From Matching to Generation: A Survey on Generative Information Retrieval  arXiv2404

*一个核心的问题：能否直接通过大模型完成文档的检索/召回？*

- 大模型并没有检索能力
- 大模型瞎编烂造的能力在检索相关问题上体现的淋漓尽致
- 大模型需要定向微调才能实现检索能力

![image-20240527100304250](imgs/image-20240527100304250.png)

#### 1. 生成式检索模型目前面临的问题

![image-20240527100758594](imgs/image-20240527100758594.png)

##### 增量学习问题

	- 文档的动态更新，大模型怎么去适配？
	- 如何处理海量的文档？
	- 如何将文档嵌入到模型中？

##### 文档标识定义

- 如何定义DockID能够让模型更轻松的记忆和泛化

##### 训练策略和模型架构

- 如何设计架构和策略来让模型更高效的记忆和泛化海量文档

##### 生成答案

- 如何通过查询到的文档高效的生成答案

#### 2. 经典工作

##### DSI (Google)

提出一种分层的文档编码方案

Transformer Memory as a Differentiable Search Index NeruIPS 2022

![image-20240527101451844](imgs/image-20240527101451844.png)

##### WebUltron (renda)

给出一种三阶段训练框架

WebUltron: An Ultimate Retriever on Webpages Under the Model-Centric Paradigm 2023 IEEE Transactions on Knowledge and Data Engineering

![image-20240527101829440](imgs/image-20240527101829440.png)

##### NCI (MicroSoft)

A Neural Corpus Indexer for Document Retrieval NeruIPS 2022

提出一种神经语料库检索器，序列到序列的网络

![image-20240527102128282](imgs/image-20240527102128282.png)

##### 跳出Sequence范式，词袋模型 (renda)

Generative Retrieval via Term Set Generation  SIGIR 2024

文档ID是序列化的形式，解码错一步则全错

提出基于词袋的方案，生成词袋的顺序构成了文档标识符

![image-20240527102221733](imgs/image-20240527102221733.png)

##### 可学习的文档标识符 (renda)

NOVO: Learnable and Interpretable Document Identifiers for Model-Based IR  CIKM 2023

现有的文档ID基于Encoder独立完成，与Decoder无关，存在Gap，提出了一种可学习的文档标识符方案

![image-20240527102858699](imgs/image-20240527102858699.png)

##### 相关性强化的生成式检索模型 （renda）

Enhancing Generative Retrieval with Reinforcement Learning from Relevance Feedback EMNLP 2023

引入基于相关性反馈的强化学习来让模型理解相关性

![image-20240527103051471](imgs/image-20240527103051471.png)

##### 生成式检索与其他生成任务的融合 （renda）

UniGen: A Unified Generative Framework for Retrieval and Question Answering with Large Language Models  AAAI 2024

![image-20240527103209757](imgs/image-20240527103209757.png)

##### CorpusLM (renda)

CorpusLM: Towards a Unified Language Model on Corpus for Knowledge-Intensive Tasks  SIGIR 2024

![image-20240527103302638](imgs/image-20240527103302638.png)



## Report 3 大模型时代的通用向量检索

北京智源研究院  刘政

### 01 什么是语义向量模型

向量模型：将任意数据转化为高维空间中稠密向量的计算机模型

重要属性：**向量相似性**要与**数据相似性**保持一致

这里的相似性计算并不严格，不受三角不等式约束

![image-20240527103615275](imgs/image-20240527103615275.png)

向量模型应用：

- 信息检索系统
- 比较数据的语义关联，数据聚类去重等
- 向量数据库：不再追求相似性最高的目标向量，近似最近邻计算节约开销

### 02 向量学习的基本模式

![image-20240527112225640](imgs/image-20240527112225640.png)

#### 1. 训练数据

向量模型的训练数据通常包含三个组成单元：查询、查询对应的关联文档以及全部的文档集合

![image-20240528102219581](imgs/image-20240528102219581.png)

向量模型的训练数据是一种相对稀缺的资源，最常用的是微软研究院发布的MS MARCO数据集，这是发布最早、规模最大、质量最好的数据集之一。

![image-20240528102608523](imgs/image-20240528102608523.png)

针对学习数据稀缺的问题，主要原因是传统的向量学习高度依赖人类的标注，大规模扩增不可能。目前一种解决方案是利用机器自动化的生产制造数据

- UltraFeedback (THU)
- Alpaca (Stanford)

#### 2. 模型

![image-20240527112454548](imgs/image-20240527112454548.png)

- 几乎所有的DNN模型都可以作为向量模型
- 但真正让向量模型变得可用的是，以Transformer为基础的预训练模型，其本质是“大”
- Transformer的网络架构可以做的足够大，让模型由足够的容量去建立强大的表达能力
- 预训练的规模可以足够大，让模型充分学习海量数据中的知识

##### 主流的预训练算法对向量检索而言是不是最优的？

- 修改模型训练目标：让embedding直接体现在训练的优化之中
- 优化embedding对输入文本语义的表达能力：先前的单个词的预测不合适，简单的转为预测目标文本的全部词汇

基于此：提出RetroMAE

RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder  EMNLP 2022

- 从任意的无标签语料，比如Wikipedia中采样一段文本，将其编码成embedding之后再将其解码出原始的文本
- 非对称的编码器解码器结构，非对称的掩码比率，使得模型输出的embedding可以更加充分的从训练数据中得到学习优化

![image-20240527113712720](imgs/image-20240527113712720.png)

**未来持续增大预训练模型的规模是一个必然的趋势，性能随着模型规模一致提升**

#### 3. 训练算法

**讲者观点是训练数据和模型的重要性要高于训练算法，尽管训练算法也同样重要**

向量学习训练算法的基本形式是对比学习，我们可以进行优化的自由度只有两个：正样本和负样本

![image-20240528103806433](imgs/image-20240528103806433.png)

##### Positive Sample

如何找到正样本？

借助一个已知的检索模型，找到关联性的候选文档，并对其进行细粒度的关联性标注，借助蒸馏技术对向量模型进行精细化的训练

![image-20240528104215321](imgs/image-20240528104215321.png)

##### Negative Sample

![image-20240528104530010](imgs/image-20240528104530010.png)

我们希望正样本可以从所有负样本中轻松分辨出来，除了正样本之外所有的文档都为负样本，但是硬件并不允许我们这样做。提出两点：

- 尽可能多的引入负样本，负样本越多，再损失上更接近与无偏

  - 批次间共享 (in-batch negative sampling)  不同设备 (cross-device negative sampling)、不同训练step (gradient-checkpoint based sampling)之间共享
  - 同样要付出更大的训练集群和更长的训练时间的代价

  ![image-20240528105112878](imgs/image-20240528105112878.png)

- 挖掘更难的负样本，从统计学的角度来看这相当于重要性采样；从损失的角度来看，更难的负样本会带来更大的梯度更新

  - 检索模型查询到与query关联度高的候选文档，都可以认为是难负样本

  ![image-20240528105709513](imgs/image-20240528105709513.png)



### 03 BGE模型的开发与实践

https://github.com/FlagOpen/FlagEmbedding/tree/master

B (BAAI) G (General) E (Embedding)

- 我们希望模型足够通用，能够再不同场景下具有适用性且性能出色
- 向量模型的通用性成为了RAG的一个瓶颈
- 我们认为**通用能力**是未来的一个主要的优化目标

通用性的维度：

- 任务场景：服务任意的应用，无论其涉及的知识理论是什么
- 语言：支持任意语言的检索诉求，实现任意语言之间的语义胡同
- 数据模态：文本、图像、语音、甚至是分子结构

最终目标是同时实现这三个维度，这是一个很困难的问题。

##### BGE v1  23.08

![image-20240528111737358](imgs/image-20240528111737358.png)

- 数据建设： 足够的、多样化的、高质量的训练数据

  - 数据质量相对较低但规模巨大： 海量的公开语料，包含大量半结构化信息
  - 数据规模较小，但质量较高，覆盖关键能力：收集了语义关联文本QA数据集；语义等价文本数据集

  ![image-20240528112157011](imgs/image-20240528112157011.png)

- 训练方法：大规模的训练+精细化的训练

  - 第一阶段强调规模化，最基础的对比学习，使得模型初步但全面的建立复杂、多样语义关系的匹配能力
  - 第二阶段强强调精细化，难例挖掘、数据增强、知识蒸馏等，建立模型对关键领域的语义匹配能力

  ![image-20240528112400273](imgs/image-20240528112400273.png)

##### Improved BGE: M3 Embedding (BGE v2) 24.02

BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation  arXiv2402

- Multi-Lingual：首要目标是打通语言壁垒，一个模型支撑不同语言的多语言检索能力以及跨语言检索能力
- Multi-Granularity：获得更长的序列范围 （8192 tokens）
- Multi-Functionality：希望一个模型集成向量检索、关键词检索重排序模块等功能

数据建设：

- type 1 ： 1.2billion 比BGEv1高出一个数量级，覆盖率100种以上的语言
- type 2 ： 语义关联、语义等价

![image-20240528113413558](imgs/image-20240528113413558.png)

训练：

- 提升算力，训练了高达67k epochs
- 引入自蒸馏技术，集成了多种检索模式

![image-20240528113648288](imgs/image-20240528113648288.png)

支持多种小语种的语言模型，被评为向量模型中的瑞士军刀

### 04 大语言模型与信息检索

**从长远来看，大语言模型势必取代搜索引擎**

取代需要解决的两大问题：

- 能够处理的上下文足够长（窗口足够长） 目前拓展长度受到系统硬件制约
- 处理长序列的能力足够强
- 成本控制

（偷听：北京大学目前再做**对于长文本的精细化检索**，实验发现检索结果会随着文档的长度的增加而降低）

##### ACL24 CFIC 建模更长的检索结果

Grounding Language Model with Chunking-Free In-Context Retrieval  ACL 2024

与人大合作的一项工作，见Report 2 Part 2

### 05 总结

- 实现通用的向量模型还有很长的路要走
- 我们认为数据基础跟模型是实现通用模型的核心要素
- 未来生成式检索会变得愈发流形

## Report 4 检索增强大模型技术探索与思考

中国科学院计算技术研究所 庞亮

![image-20240528115228871](imgs/image-20240528115228871.png)

目前面临的核心问题：

- 不能准确的获得知识 （检索视角）
- 不能准确的选择知识 （大模型视角）
- 知识之间的干扰 （交互视角）

从四个角度介绍计算所近期的工作

### 01 检索视角下的检索增强

什么是适合大语言模型的检索增强的信息检索？

- 应用范围广，任务种类多，对跨领域跨任务泛化性要求高
- 推理开销大，上下文空间有限，对排序精度和鲁棒性要求高

研究现状：

![image-20240528134035025](imgs/image-20240528134035025.png)

#### BERM：训练匹配的平衡可提取表示提高密集检索的泛化能力

BERM: Training the Balanced and Extractable Representation for Matching to Improve Generalization Ability of Dense Retrieval  ACL 2023

- 动机： 大部分稠密向量检索算法效果再数据集之外的场景泛化性能非常差

![image-20240528134517286](imgs/image-20240528134517286.png)

- 提出了匹配表示概念
- 提出了可泛化的稠密向量检索模型再训练时的两个要求

![image-20240528134615013](imgs/image-20240528134615013.png)

#### Match-Prompt：通过提示学习提高神经文本匹配的多任务泛化能力

Match-Prompt: Improving Multi-task Generalization Ability for Neural Text Matching via Prompt Learning  CIKM 2022

![image-20240528134856756](imgs/image-20240528134856756.png)

![image-20240528134925696](imgs/image-20240528134925696.png)

#### NIR-Prompt：多任务可泛化神经信息检索训练框架

NIR-Prompt: A Multi-task Generalized Neural Information Retrieval Training Framework  ACM Transactions on Information Systems 2023

![image-20240528135213384](imgs/image-20240528135213384.png)

#### GenRT：检索增强生成的列表感知重排序-阶段联合模型

List-aware Reranking-Truncation Joint Model for Search and Retrieval-augmented Generation  WWW 2024

![image-20240528135241409](imgs/image-20240528135241409.png)

### 02 大模型视角下的检索增强

大模型如何鲁棒的对抗输入的噪音知识，并在参数内外知识之间做出选择

- 有监督指令微调：在领域特定数据集上构造检索-问题-答案三元组，利用构造的有监督三元组进行指令微调，教会大模型如何使用检索到的文档
- 强化学习：强化学习对齐大语言模型在使用检索文档上的偏好
- 模型蒸馏：使用更强大的模型作为教师模型微调生成器
- 生成器与检索器协同微调：最小化检索器分布于与LLM偏好之间的KL散度以及最大化给定检索增强指令情况下正确答案的可能性

#### Info-RAG：用于检索增强生成的大型语言模型的无监督信息细化训练

Unsupervised Information Refinement Training of Large Language Models for Retrieval-Augmented Generation  ACL 2024

![image-20240528140241985](imgs/image-20240528140241985.png)

### 03 交互视角下的检索增强

大模型与信息检索如何高效交互从而鲁棒的解决复杂问题？

交互框架：

- 基于工具调用 ToolFormer
- 基于复杂问题分解  Self-Ask  DSP

#### Search-in-the-Chain: 面向知识密集型任务的交互式检索增强大型语言模型

Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks  WWW  2024

![image-20240528140802334](imgs/image-20240528140802334.png)

在面对复杂的需要多跳知识密集型的问题时，现存的检索增强的交互框架存在以下问题：

- 检索与大模型的交互打破了大模型连贯的推理链，使其在每次推理时仅能解决“局部”节点的问题。
- 在每个节点都直接将文档提供给大模型，存在误导大模型的风险，也增加的大模型的推理开销。
- 推理的方向不能动态调整，且输出的内容无法溯源，缺乏可验证性。

![image-20240528140916368](imgs/image-20240528140916368.png)

### 04 信息回路视角下的检索增强

大模型生成的内容被混入检索的语料库时，将如何影响信息检索的表现？

![image-20240528141017768](imgs/image-20240528141017768.png)

#### 大语言模型可能主导信息获取：神经检索器偏向大语言模型生成的文本

LLMs may Dominate Information Access: Neural Retrievers are Biased Towards LLM-Generated Texts  KDD 2024

利用改写的方式把真实语料库中的文本转化成大模型生成的文本，并将真实文本和生成的文本混合做为评测基准

![image-20240528141531077](imgs/image-20240528141531077.png)



#### 看不见的相关性偏差：文本图像检索模型更喜欢人工智能生成的图像

Invisible Relevance Bias: Text-Image Retrieval Models Prefer AI-Generated Images  SIGIR 2024

采用先过采样生成，后筛选的策略，选择和真实图片语义最一致的生成图片

![image-20240528141639718](imgs/image-20240528141639718.png)

## Report 5  When Search Engine Meets LLMs: Opportunities and Challenges
微软亚洲研究院  王亮

### 01 LLMs如何帮助现有的搜索技术栈

**信息检索最根本的问题是表示学习**

**表征学习最重要的是尺度定律[openai 2020]**

![image-20240528160954077](imgs/image-20240528160954077.png)

![image-20240528161006955](imgs/image-20240528161006955.png)

#### 1. Generative Retrieval 生成式检索

##### Differentiable Search Index (DSI)

Transformer Memory as a Differentiable Search Index, 2022

![image-20240528154438387](imgs/image-20240528154438387.png)

##### Generative Retrieval - SEAL

Autoregressive Search Engines: Generating Substrings as Document Identifiers, 2022

![image-20240528154500926](imgs/image-20240528154500926.png)

##### Limitations of Generative Retrieval

How Does Generative Retrieval Scale to Millions of Passages?, 2023

![image-20240528154614482](imgs/image-20240528154614482.png)

#### 2. Embedding-based Dense Retrieval 基于向量模型的密集检索 

![image-20240528155437686](imgs/image-20240528155437686.png)

如何增强密集检索？

![image-20240528155631344](imgs/image-20240528155631344.png)

ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT, 2020 

RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering, 2020 

Approximate Nearest Neighbor Negative Contrastive Learning for Dense Text Retrieval, 2020 

Adversarial Retriever-Ranker for dense text retrieval, 2021 

Text Embeddings by Weakly-Supervised Contrastive Pre-training, 2022

SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval, 2022 

RetroMAE: Pre-Training Retrieval-oriented Language Models Via Masked Auto-Encoder, 2022

##### Knowledge distillation from re-ranker

RocketQAv2: A Joint Training Method for Dense Passage Retrieval and Passage Re-ranking, 2021

![image-20240528155856521](imgs/image-20240528155856521.png)

##### Continual pre-training 

SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval, 2022

![image-20240528155929871](imgs/image-20240528155929871.png)

Text Embeddings by Weakly-Supervised Contrastive Pre-training, 2022

![image-20240528155957780](imgs/image-20240528155957780.png)

##### The Importance of Large Batch Size

Text Embeddings by Weakly-Supervised Contrastive Pre-training, 2022

![image-20240528160057200](imgs/image-20240528160057200.png)

##### GradCache

Scaling deep contrastive learning batch size under memory limited setup, 2021

![image-20240528160118452](imgs/image-20240528160118452.png)

![image-20240528160135248](imgs/image-20240528160135248.png)

#### 3. LLMs + IR 大语言模型+信息检索

##### RankLLaMA

Fine-Tuning LLaMA for Multi-Stage Text Retrieval, 2023

![image-20240528161115573](imgs/image-20240528161115573.png)

##### SGPT

SGPT: GPT sentence embeddings for semantic search, 2022

![image-20240528161143463](imgs/image-20240528161143463.png)

##### E5 Mistral

Improving text embeddings with large language models, 2024

##### GritLM: Unifying Text Generation and Embeddings

Generative representational instruction tuning, 2024

#### 4. Challenges to Deploy LLM-based Embeddings 部署基于LLM的向量模型的挑战

- 推理成本
  - 半精度推理
  - 更好的推理实现 （FlashAttention-2）
  - 蒸馏到更小的模型
- 存储成本
  - 向量量化

##### Matryoshka Embeddings

Matryoshka Representation Learning, 2022

![image-20240528161706588](imgs/image-20240528161706588.png)



### 02 搜索引擎如何增强LLMs

LLMs的缺点：

- 无法获取最新事件
- 缺乏专业领域知识
- 微调注入新知识困难

#### 1. Rag Pipeline

![image-20240528162041546](imgs/image-20240528162041546.png)

##### KNN-LM

Generalization through Memorization: Nearest Neighbor Language Models, 2019

![image-20240528162115789](imgs/image-20240528162115789.png)

##### RETRO

Improving language models by retrieving from trillions of tokens, 2021

![image-20240528162143435](imgs/image-20240528162143435.png)

##### REPLUG

Replug: Retrieval-augmented black-box language models, 2023

![image-20240528162206955](imgs/image-20240528162206955.png)

#### 2. RAG Agent

##### WebGPT

WebGPT: Browser-assisted question-answering with human feedback, 2021

![image-20240528162821018](imgs/image-20240528162821018.png)

##### Self-RAG

Self-RAG: Learning to retrieve, generate, and critique through self-reflection, 2023

![image-20240528162855349](imgs/image-20240528162855349.png)

#### 3. RAG 与长上下文LLMs

##### Fusion-in Decoder (FiD)

Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering, 2020

![image-20240528163006167](imgs/image-20240528163006167.png)

##### Position Interpolation

Extending context window of large language models via positional interpolation, 2023

![image-20240528163028450](imgs/image-20240528163028450.png)

##### PoSE

Pose: Efficient context window extension of llms via positional skip-wise training, 2023

![image-20240528163046285](imgs/image-20240528163046285.png)

#### 4. 面临挑战

##### 长上下文理解

Lost in the middle: How language models use long contexts, 2023

![image-20240528163154265](imgs/image-20240528163154265.png)

##### 推理效率

Inference with reference: Lossless acceleration of large language models, 2023

![image-20240528163211080](imgs/image-20240528163211080.png)

##### 来源归属

Evaluating verifiability in generative search engines, 2023

![image-20240528163243718](imgs/image-20240528163243718.png)

### 03 LLMs会取代搜索引擎么

目前的障碍：

- 高效持续的学习新知识
- 幻觉问题
- 推理的成本和延迟

#### 谷歌的一项提案

Rethinking Search: Making Domain Experts out of Dilettantes, 2021

![image-20240528163326785](imgs/image-20240528163326785.png)

#### Large Search Model 微软的提案

Large Search Model: Redefining Search Stack in the Era of LLMs, 2023

![image-20240528163403763](imgs/image-20240528163403763.png)

### 04  结论

####  LLMs如何帮助现有的搜索技术栈？

- 生成式检索
- 利用LLM进行文本检索和排名
- 多角度进行数据合成

####  搜索引擎如何增强LLMs？

- 检索增强生成
- 具有检索功能的Agent

#### LLMs会取代搜索引擎么

- 在可预见的未来中，LLM和搜索引擎可能s

## Report 6. LLM-Based Tool Learning and Autonomous Agents

中国人民大学 高瓴人工智能学院 林衍凯

Survey: Tool Learning with Foundation Models arXiv2304

![image-20240526225617851](imgs/image-20240526225617851.png)

- 控制器提供可行的计划来满足用户的请求
- 工具集合为一系列拥有不同功能的集合
- 环境提供工具操作的平台
- 感知者向控制器总结反馈信息

![image-20240526230159979](imgs/image-20240526230159979.png)

### Intent Understanding: understand the user task intent

意图理解，理解用户的任务意图

难点：理解用户的模糊指令

将任务传递给下级执行之前，agent应当主动且明确的向用户询问缺失的细节

![image-20240526231522480](imgs/image-20240526231522480.png)

一个例子：

![image-20240526231555046](imgs/image-20240526231555046.png)

### Planning: divide the user query into sub-tasks

将用户的查询分解为多个子问题来分别处理

#### Chain of Thought (CoT)

思维链方法：给出推导过程的思维过程

![image-20240526232354418](imgs/image-20240526232354418.png)

#### Tree of Thought (ToT)

思维树方法：给出推导过程的搜索树

![image-20240526232451604](imgs/image-20240526232451604.png)

#### ReAct  2023 ICLR

ReAct: Synergizing Reasoning and Acting in Language Models

CoT推理帮助模型更新行动计划以及处理异常，而行动允许它与外部源（例如知识库或环境）进行交互，以收集更多信息

![image-20240526232712864](imgs/image-20240526232712864.png)

#### DFSDT 深度优先搜索的决策树

ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs arXiv2307

训练了 API 检索器来为每条指令推荐适当的 API

![image-20240526233933016](imgs/image-20240526233933016.png)

### Human-Agent Collaboration 人类-智能体协作

有些问题全部由智能体来代理性能并不佳，但是将其中很小一部分交给人类完成性能能够得到大幅提升，引入人机协作问题

![image-20240526234452934](imgs/image-20240526234452934.png)

$\lambda$来控制人类的参与程度

![image-20240526234809869](imgs/image-20240526234809869.png)

### Tool Use: use the appropriate tool to solve sub-task

#### 模仿学习：一个最简单的学习范式。

通过记录人类使用工具的行为数据，让大模型来模拟人类的行为来了解工具

![image-20240526234956578](imgs/image-20240526234956578.png)

##### WebGPT

WebGPT: Browser-assisted question-answering with human feedback arXiv2112

- 模仿类使用搜索引擎的行为
- 监督微调 + 强化学习
- 只需 6,000 条标注数据

![image-20240526235241742](imgs/image-20240526235241742.png)

##### WebCPM: Chinese WebGPT

Interactive web search for Chinese long-form question answering. ACL 2023

中文版的WebGPT

![image-20240526235319825](imgs/image-20240526235319825.png)

在每个步骤中，搜索模型都会执行操作以收集证据，并将其发送给大模型以生成答案

![image-20240526235402358](imgs/image-20240526235402358.png)

##### WebShop

agent学习网上购物

![image-20240526235530148](imgs/image-20240526235530148.png)

##### GUIAgent

学习操作GUI工具，与VLM模型结合

![image-20240526235609456](imgs/image-20240526235609456.png)

#### 教程学习：让模型阅读工具手册来学习

OpenAI的大模型都有很强的zero-shot能力，能够理解手册的内容

Zero-shot & Few-shot 的例子

![image-20240527000011510](imgs/image-20240527000011510.png)

##### ToolBench

https://github.com/OpenBMB/ToolBench

- 集成了来自RapidAPI超过16000个API

  - 选取了16, 000多个高质量API
  - 涵盖了49个类别

- 支持单工具或多工具的调用

  - 简单的api指令集合
  - chatgpt自动生成指令，可能包括一个或多个api

  ![image-20240527000843773](imgs/image-20240527000843773.png)

- 支持复杂的推理任务

![image-20240527000512994](imgs/image-20240527000512994.png)

#### 一个模型学习使用工具的例子：VPT

Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos NeurIPS 2022

### Memory: manage the working history

#### Short-Term Memory  短期记忆

短时记忆通常是通过上下文学习实现的，记忆信息直接写入prompt中

![image-20240527001410443](imgs/image-20240527001410443.png)

#### Short-Term Memory + Long-Term Memory 短期+长期记忆

外部记忆存储+检索外部记忆+短期记忆

![image-20240527001914512](imgs/image-20240527001914512.png)

#### 如何存储长期记忆

##### 直接写入Raw Text

![image-20240527002121815](imgs/image-20240527002121815.png)

##### 编码后写入

![image-20240527002154719](imgs/image-20240527002154719.png)

#### 存储策略

- 基于LLM合并相同的记忆
- 遵循先进先出的原则，最早的记忆会首先被覆盖

![image-20240527002312355](imgs/image-20240527002312355.png)

#### 人类对记忆能够自证和评估，我们希望模型也可以

##### Self-summarization

![image-20240527002749521](imgs/image-20240527002749521.png)

##### Self-verification

![image-20240527002812345](imgs/image-20240527002812345.png)

#### Reflexion NeurIPS 2023

利用语言反馈信号强化agent，从之前的失败中吸取教训

![image-20240527002933155](imgs/image-20240527002933155.png)

#### Agent的安全性讨论

- Agent本身可能会被注入目的性的引导，例如用来购物的agent可能会被操控倾向于选择某类商品，这对用户是很难以察觉的
- Agent调用的API本身安全性是否能够得到保障？有些工具本身就存在一些不安全的因素
- Agent调用某个工具的原因是一个黑盒，这对一些敏感场景有风险（自动驾驶 医疗系统）



## Report 7. 工业界专场

*百川 技术负责人 方琨*

*智谱 解决方案专家 冯小平*

*Jina AI 联合创始人兼CEO 王楠* 

### 主要观点

- 意图理解 
  - 更多的去解读用户意图，适应用户意图
  - 针对特定的应用场景，将大模型从通用$\rightarrow$专用
- 大规模处理
  - 近期GPT4出现连接云盘的接口，目前只能处理上传一个文件
  - 但这是一个未来趋势，能够整合云盘中大量的结构化、非结构化数据
- 企业的tool calling就绪度目前还差很多
- 很多时候更加专注延迟等用户体验的指标

### 企业大模型的一个简单框架

![image-20240527093804005](imgs/image-20240527093804005.png)

### 一个很好的样例：

![image-20240527094009819](imgs/image-20240527094009819.png)

#### Query改写

![image-20240527094227419](imgs/image-20240527094227419.png)

#### 等价Query：并行多查询

![image-20240527094310111](imgs/image-20240527094310111.png)

![image-20240527094322616](imgs/image-20240527094322616.png)

#### 更抽象的Query：回撤

![image-20240527094348297](imgs/image-20240527094348297.png)

![image-20240527094400567](imgs/image-20240527094400567.png)

#### 更具体的Query：任务分解

![image-20240527094428556](C:/Users/bryce/AppData/Roaming/Typora/typora-user-images/image-20240527094428556.png)

![image-20240527094512190](imgs/image-20240527094512190.png)

#### 根据Query生成Document：HyDE

使用查询到的文档+Query进行查询

![image-20240527094607894](imgs/image-20240527094607894.png)

![image-20240527094631113](imgs/image-20240527094631113.png)

#### 路由控制器

- 意图识别
- 根据用户意图决定召回路径

![image-20240527094950063](imgs/image-20240527094950063.png)

![image-20240527095005866](imgs/image-20240527095005866.png)

### Jina AI的RAG工具

####  jina-embeddings-v2

23.10发布，全球第一个支持8k输入长度的开源向量模型

#### jina-colbert-v1

第一款支持8k长度的colbert模型

#### jina-reranker-v1

基于jina bert v2 支持8k上下文输入

#### jina-clip-v1

正在开发...

#### AIR-Bench

自动化多样信息检索评测基准

### 思考和讨论

![image-20240527095543319](imgs/image-20240527095543319.png)

**如果未来LLM能够精确的召回记忆，那么RAG将不再被需要**。

![image-20240527095750548](imgs/image-20240527095750548.png)
