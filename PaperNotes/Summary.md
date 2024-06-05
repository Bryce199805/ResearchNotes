

Revisiting Label Smoothing and Knowledge Distillation Compatibility What was Missing?

引用的两篇 没看

Is label smoothing truly incompatible with knowledge distillation: An empirical study.

When does label smoothing help? 

**我们称经过softmax的logits为预测logits方法，未经过softmax的称之为特征logits**

## Logits Distillation

| 序号 | 年份 | 出处          | 题目                                                         | 关键点                                                       | 目标                     |
| :--: | :--: | :------------ | :----------------------------------------------------------- | :----------------------------------------------------------- | ------------------------ |
|  1   | 2014 | NeurIPS       | Distilling the Knowledge in a Neural Network                 | <li> 首次提出蒸馏的概念。 <br><li> 提出了使用软标签来训练学生网络<br><li> 提出了温度系数的概念来加强概率较低的标签概率对于模型的影响 | 预测logits               |
|  2   | 2014 | NeurIPS       | Do Deep Nets Really Need to be Deep?                         | <li> 使用简单网络模仿已经训练好的复杂网络的logit输出<br><li> 在隐藏层的学习中引入了矩阵分解来加快模型的学习 | 特征logits               |
|  3   | 2016 | arXiv         | Deep Model Compression: Distilling Knowledge from Noisy Teachers | <li> 对教师输出的logits添加噪声来模拟同专业的多个教师<br><li> 添加噪声相当于正则化项的作用 | 预测logits               |
|  4   | 2017 | NeurIPS       | Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results | <li>提出平均模型权重，称之为均值教师<br><li>提出使用学生模型的指数移动平均EMA权重，在每一步之后聚合信息<br><li>这是对2017ICLR时序集成模型的改进 | 预测logits               |
|  5   | 2018 | AAAI          | Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net | <li> 提出一个共享低级别层参数的师生网络架构<br><li>发现对logits进行蒸馏要优于经过softmax后的结果<br><li>提出了梯度阻断模块，阻止轻量网络的梯度对加速网络的负优化 | 特征logits               |
|  6   | 2018 | CVPR          | Deep Mutual Learning                                         | <li>提出互学习的概念，让多个学生模型之间互相学习<br><li>给出了模型逐个蒸馏与集成蒸馏的方案<br><li>证明了具有高熵的后验概率模型能训练出更鲁棒的模型结构 | 预测logits               |
|  7   | 2018 | PMLR          | Born Again Neural Networks (BANs)                            | <li> 序列化建模，前一个模型为后一个模型的老师<br><li>将所有学生模型集成得到最终的学生模型 | 预测logits               |
|  8   | 2019 | AAAI          | Training Deep Neural Networks in Generations: A More Tolerant Teacher Educates Better Students | <li>  考虑了次要类别的影响，视觉上相似的物体同样包含着有效的信息<br><li>引入了类别相似性的知识，计算主类与前k个类别置信度差异 | 预测logits               |
|  9   | 2019 | CVPR          | Relational Knowledge Distillation (RKD)                      | <li> 提出关注样本之间关系<br><li> 提出距离和角度的蒸馏损失来惩罚模型之间的结构差异 | 特征logits；实例关系     |
|  10  | 2020 | PMLR          | Self-supervised Label Augmentation via Input Transformations | <li> 线性可分的任务经过某些数据增强手段之后可能不在线性可分<br><li> 学习一个原始标签与增强标签联合分布的任务 | 预测logits<br>数据增强   |
|  11  | 2021 | ICCV          | Densely Guided Knowledge Distillation using Multiple Teacher Assistants | <li> 解决使用辅助模型弥合师生模型差距时的错误累计问题<br><li> 借鉴DenseNets思想提出了DGKD框架<br><li> 借鉴Dropout提出随机丢弃缓解过拟合 | 特征logits               |
|  12  | 2021 | ICLR          | Knowledge Distillation via Softmax Regression Representation Learning | <li>专注于教师分类器的学习<br><li> 一项损失匹配特征图特征，另一项师生logits都使用教师分类器匹配其输出 | 预测logits<br>倒数第二层 |
|  13  | 2021 | Image Process | ResKD: Residual-Guided Knowledge Distillation                | <li> 师生模型之间的差距可以作为知识指导学生<br><li> 通过NAS搜索得到残差学生，提出能量熵来设定停止条件<br><li>对难度不同的样本提出自适应推理加速 | 特征logits               |
|  14  | 2022 | CVPR          | Knowledge Distillation with the Reused Teacher Classifier    | <li> 直接复用教师模型的分类器来推理<br><li> 对分类器前的logit简单投影对齐后与教师模型学习 | 特征logits               |
|  15  | 2022 | ICLR          | Bag of Instances Aggregation Boosts Self-Supervised Distillation | <li> 提出袋结构来利用预训练的MoCo的知识<br><li> 将相似的数据装入一个袋中<br><li> 提出样本内损失蒸馏和样本间损失蒸馏 | 特征logits<br>实例关系   |
|  16  | 2023 | CVPR          | Multi-level Logit Distillation                               | <li> 针对logits蒸馏提出多层次对齐方式<br><li> 从实例级、批次级和类别集对教师模型对齐 | 预测logits<br>实例关系   |



## Feature Distillation

| 序号 | 年份         | 出处          | 题目                                                         | 关键点                                                       | 目标                             |
| ---- | ------------ | ------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | -------------------------------- |
| 1    | 2015         | ICLR          | FitNets: Hints For Thin Deep Nets                            | <li> 训练一个比教师模型更深但更细的网络进行模型压缩<br><li> 选择教师网络的隐藏层作为提示层，学生网络某个层作为引导层，引导层从提示层进行学习 | 中间层                           |
| 2    | 2016         | CIKM          | Distilling Word Embeddings: An Encoding Approach             | <li> 提出一个词嵌入的蒸馏，将大嵌入知识提取到小的词嵌入中<br><li> 小型嵌入会丢失某些句法信息而使用蒸馏这部分信息则会被保留 | 词嵌入                           |
| 3    | 2016         | ICLR          | Net2Net: Accelerating Learning via Knowledge Transfer        | <li> 将知识从先前的网络转移到新的更深或更宽的网络来加速实验<br><li> 给出一套权重转移的计算方法 | 初始化                           |
| 4    | 2017         | arXiv         | Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks (KPN) | <li> 提出一种知识投影网络(KPN) 分两阶段优化<br><li> 定义投射层，第一阶段学生网络上半部分学习对应教师投射层之前的输出，学生下半部分与投射层通过交叉熵指导学习<br><li> 第二阶段使用第一阶段的结果作为初始化，对整个学生网络优化<br><li> 提出投影路径迭代剪枝来选择最佳投影层位置 | 中间层                           |
| 5    | 2017         | CVPR          | A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning (FSP) | <li> 提出表示中间求解过程流程的FSP矩阵，由两个隐藏层计算<br><li> 第一阶段对学习教师模型的FSP矩阵，第二阶段通过任务损失优化 | 中间层                           |
| 6    | 2017<br>2019 | arXiv<br>ICLR | Like What You Like: Knowledge Distill via Neuron Selectivity Transfer (NST) | <li> 提出最小化师生模型特征图分布之间的最大均值差异MMD来匹配神经元的选择性分布<br><li> 通过核函数来进行映射函数的辅助计算 | 中间层<br>激活模式               |
| 7    | 2017         | ICLR          | Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer (AT) | <li> 提出基于注意力图的转移方式<br><li> 对师生模型分组，基于组来传递注意力图知识<br><li> 给出了三种较常用的注意力图计算方式 | 中间层<br>注意力                 |
| 8    | 2018         | AAAI          | DarkRank: Accelerating Deep Metric Learning via Cross Sample Similarities Transfer | <li> 提出了知识转移的样本交叉相似性 <br><li> 给出一个打分函数，计算一个miniBatch中样本之间的相似性并给出相应的损失计算 | 中间层<br>样本相似性             |
| 9    | 2018         | ECCV          | Learning Deep Representations with Probabilistic Knowledge Transfer | <li> 提出一种基于概率的知识迁移方法，让学生模仿教师的特征空间<br><li>提出通过核函数来辅助构建教师模型的特征空间<br><li>给出了信息论角度的分析论证 | 中间层<br>倒数第二层<br>信息论   |
| 10   | 2018         | ECCV          | Quantization Mimic: Towards Very Tiny CNN for Object Detection | <li> 将量化方法与蒸馏方法相结合<br><li> 让学生特征图维度对齐，对师生模型均匀量化后的特征图进行蒸馏学习<br><li> 量化相当于正则化的作用，并给出了不同量化方法的分析 | 中间层                           |
| 11   | 2018         | NeurIPS       | Moonshine: Distilling with Cheap Convolutions                | <li> 对学生架构的设计，保持原始架构不变，替换更便宜的卷积块<br><li> 提出分组卷积，降低开销<br><li>提出瓶颈块进一步降低参数数量 | 中间层                           |
| 12   | 2018         | NeurIPS       | Paraphrasing Complex Network: Network Compression via Factor Transfer (FT) | <li> 提出释义器和翻译器架构，前者解释教师特征，后者帮助学生翻译特征内容<br><li> 分别添加到师生模型的最后一组卷积之后来处理特征图<br><li>通过p范数来匹配翻译器和释义器的输出 | 中间层<br>倒数第二层             |
| 13   | 2018         | PMLR          | Knowledge Transfer with Jacobian Matching                    | <li> 提出使用雅可比矩阵匹配师生模型之间的知识转移 <br><li> 雅可比矩阵匹配等价于在训练过程中对添加噪声的软标签进行匹配<br><li>通过泰勒公式对神经网络进行仿射逼近进行了证明 | 中间层                           |
| 14   | 2019         | AAAI          | Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons （AB） | <li> 提出学习教师的激活边界信息<br><li> 给出了线性层和卷积层的激活信息传递方法<br><li>提出一个替代函数来处理激活边界离散无法优化的问题 | 中间层<br>激活模式               |
| 15   | 2019         | arXiv         | MSD: Multi-Self-Distillation Learning via Multi-Classifiers within Deep Neural Networks | <li>提出对具有块结构的网络进行采样分支扩充<br><li> 提出多重自蒸馏，由标签损失，logit损失和特征损失组成<br><li> 特整层的模仿学习在后期会影响网络性能，引入余弦退火来削弱后期特征损失权重 | 中间层<br>logits                 |
| 16   | 2019         | CVPR          | Knowledge Distillation via Instance Relationship Graph       | <li> 提出实例关系图来利用样本间实例关系<br><li> 利用实例特征、实例关系和特征空间变换三种知识 | 中间层<br>实例关系<br>logits     |
| 17   | 2019         | CVPR          | Variational Information Distillation for Knowledge Transfer (VID) | <li> 给出了知识转移的信息论解释<br><li> 知识转移表述为师生网络之间的互信息最大化<br><li> 通过最大化变分下界来近似分布帮助互信息计算 | 中间层<br>信息论                 |
| 18   | 2019         | ICCV          | Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation | <li> 网络分组自蒸馏方法<br><li> 通过交叉熵 KL损失 和 Hint损失来优化模型 | 中间层<br>特征logits             |
| 19   | 2019         | ICCV          | Corrlation Congruence for Knowledge Distillation (CCKD)      | <li> 不仅考虑实例一致性，还考虑样本相关一致性<br><li>  通过内核的方法捕获样本之间的相关性<br><li> 提出一个采样器来采样计算样本相关性 | 中间层<br>实例关系               |
| 20   | 2019         | ICCV          | A Comprehensive Overhaul of Feature Distillaion              | <li> 详细对比了具有代表性的六种蒸馏方法<br><li> 修改ReLU结构保存教师信息<br><li> 给出一种新的距离度量 <br><li> 分析修改BN层 | 中间层                           |
| 21   | 2019         | ICCV          | Similarity-Preserving Knowledge Distillation (SP)            | <li> 相似样本引发相似的激活模式<br><li> 利用样本相似性传递激活模式 | 中间层<br>激活模式               |
| 22   | 2020         | ICLR          | Contrastive Representation Distillation (CRD)                | <li> 引入对比损失项考虑目标之间的依赖关系<br><li> 最大化教师和学生表示之间的互信息下界 | 中间层<br>倒数第二层<br>实例关系 |
| 23   | 2021         | AAAI          | Cross-Layer Distillation with Semantic Calibration           | <li> 处理中间层语义不匹配<br><li> 提出使用相似度矩阵和注意力机制来进行中间层关联 | 中间层<br>Attention              |
| 24   | 2021         | CVPR          | Distilling Knowledge via Knowledge Review                    | <li> 利用跨层级的知识<br><li> 提出让学生网络的高阶段层学习教师网络的低层特征<br><li>提出ABF注意力融合结构和HCL金字塔上下文损失 | 中间层                           |
| 25   | 2021         | ICCV          | Exploring Inter-Channel Correlation for Diversity-preserved Knowledge Distillation | <li> 提出ICC通道间相似性<br><li> 提出计算特征图通道相似性的相似性矩阵<br><li> 对较大的特征图提出分组计算聚合 | 中间层<br>倒数第二层<br>         |
| 26   | 2021         | IJCAI         | Hierarchical Self-supervised Augmented Knowledge Distillation | <li> 基于对比的学习会损害模型对原任务的表征能力<br><li>基于先前联合标签增强自蒸馏方法，将其运用到传统蒸馏中 | 中间层<br>数据增强               |
| 27   | 2022         | CVPR          | Knowledge Distillation via the Target-aware Transformer (TaT) | <li> 针对特征图语义不匹配问题提出TaT，利用教师特征对学生语义特征进行重建<br><li> 针对大特征图提出分组蒸馏和锚点蒸馏<br><li> 分组得到局部特征，锚点蒸馏得到全局特征 | 中间层<br>                       |
| 28   | 2023         | CVPR          | DisWOT: Student Architecture Search for Distillation Without Training | <li> 提出基于NAS的学生模型搜索策略<br><li>考虑类别样本之间的相似性，获得最佳学生模型 | 中间层<br>NAS                    |
| 29   | 2023         | CVPR          | Class Attention Transfer Based Knowledge Distillation        | <li> 提出类别注意力转换调整类别注意力图生成位置<br><li> 按照类别生成类别注意力图，更具有可解释性 | 中间层<br>可解释性               |
| 30   | 2024         | CVPR          | FreeKD: Knowledge Distillation via Semantic Frequency Prompt | <li>首次在蒸馏学习中利用频率域的知识<br><li>提出使用Prompt微调教师模型引入频率域知识<br><li>提出利用连续两层之间的相关矩阵作为频率损失的权重 | 中间层<br>频率域                 |
|      |              |               |                                                              |                                                              |                                  |
|      |              |               |                                                              |                                                              |                                  |
|      |              |               |                                                              |                                                              |                                  |
|      |              |               |                                                              |                                                              |                                  |
|      |              |               |                                                              |                                                              |                                  |
|      |              |               |                                                              |                                                              |                                  |



## Optimization

| 序号 | 年份 | 出处    | 题目                                                         | 关键点                                                       | 目标                                 |
| ---- | ---- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------ |
| 1    | 2019 | ICCV    | On the Efficacy of Knowledge Distillation                    | <li> 大量实验说明<br><li> 蒸馏损失后期会损害模型训练<br><li> 提出提前停止 | 蒸馏训练优化                         |
| 2    | 2020 | AAAI    | Improved Knowledge Distillation via Teacher Assistant (TAKD) | <li> 师生差距过大会导致学生模型性能下降<br><li> 引入助教帮助学生模型学习，弥合性能差距 | 蒸馏流程<br>师生差距                 |
| 3    | 2020 | CVPR    | Revisiting Knowledge Distillation via Label Smoothing Regularization | <li> 大量实验说明<br><li> 较弱的学生去教授老师，训练不佳的老师教授比他性能更好的学生，都能带来性能提升<br><li> 暗知识不仅包括类别之间的相似性，还包括对学生培训的正则化<br><li> 分析KD与标签平滑之间的关系，提出Tf-KD | 蒸馏与标签平滑                       |
| 4    | 2021 | ICLR    | UnDistillable: Making A Nasty Teacher That Cannot Teach Students | <li> 引入对抗性思想保护模型<br><li> 最大化教师模型与预训练模型的KL散度来尽可能的获得释放错误信号的能力<br><li> 通过交叉熵保证教师模型本身的正确率 | 蒸馏保护<br>                         |
| 5    | 2021 | NeurIPS | Does Knowledge Distillation Really Work?                     | <li> 大量实验说明<br><li> 从多个角度实验验证问题所在<br><li> 最终认定是优化方法限制了模型收敛 | 蒸馏不佳原因<br>蒸馏优化             |
|      | 2021 | ACL     | Annealing Knowledge Distillation                             | <li>第一阶段提出一个退火蒸馏损失来模仿教师，用MSE取代KL散度，并通过退火函数定义动态温度<br><li>第二阶段通过交叉熵使用硬标签来对学生模型微调 | 温度系数<br>师生差距                 |
| 6    | 2022 | CVPR    | Decoupled Knowledge Distillation                             | <li> 传统logits蒸馏性能不佳是优于某些因素耦合<br><li> 提出解耦方案，分为针对目标类的二元蒸馏和针对非目标类的多类别蒸馏 | logits蒸馏<br>优化解耦               |
| 7    | 2022 | CVPR    | Self-Distillation from the Last Mini-Batch for Consistency Regularization | <li> 处理先前自蒸馏方法即时信息丢失、额外内存消耗、并行化程度低<br><li> 提出最后一个小批量自蒸馏DLB，只存储上一批次数据产生的软标签 | 蒸馏训练优化                         |
| 8    | 2022 | ECCV    | A Fast Knowledge Distillation Framework for Visual Recognition (FKD) | <li> 针对蒸馏训练需要额外对教师模型推理引入额外开销<br><li> 在教师模型推理阶段保存数据增强及输出结果在训练学生时复用 | 蒸馏训练优化<br>教师推理开销         |
| 9    | 2022 | ECCV    | Knowledge Condensation Distillation (KCD)                    | <li> 认为对全部知识蒸馏是冗余的<br><li> 提出在线全局价值估计每个知识点的价值<br><li> 将知识点分为强中弱三个档次分别处理 | 蒸馏训练优化<br>知识筛选             |
| 10   | 2022 | NeurIPS | Knowledge Distillation from A Stronger Teacher               | <li> 学生应当更关注教师输出的相对排名而不是确切概率<br><li> 提出类内相似性和类间相似性<br><li> 提出使用皮尔森距离作为新的距离度量 | 蒸馏度量优化                         |
| 11   | 2023 | AAAI    | Curriculum Temperature for Knowledge Distillation (CTKD)     | <li> 教育应当遵循由易到难的过程，提出使用温度系数调整任务难度 <br><li> 提出使用GAN的思想学习温度系数<br><li> 给出了两种可学习温度模块 | 蒸馏训练优化<br>温度系数学习         |
| 12   | 2023 | CVPR    | Generalization Matters: Loss Minima Flattening via Parameter Hybridization for Efficient Online Knowledge Distillation | <li> 在线知识蒸馏，寻找平坦最优解<br><li> 提出混合权重模型HWM<br><li> 提出融合策略按照计划利用混合模型更新学生模型 | 蒸馏泛化性<br>平坦最优解             |
| 13   | 2023 | ICCV    | Automated Knowledge Distillation via Monte Carlo Tree Search | <li> 总结先前工作提出一个搜索空间<br><li> 通过蒙特卡洛树搜索寻找最佳蒸馏策略<br><li> 提出离线推理、随机掩码稀疏蒸馏、提前停止来加速搜索 | 最佳蒸馏方案搜索                     |
| 14   | 2023 | ICCV    | DOT: A Distillation-Oriented Trainer                         | <li> 认为任务损失与蒸馏损失发生冲突，退化为多任务学习<br><li> 调整优化器动量值让蒸馏损失占据主导 | 蒸馏训练优化<br>损失冲突             |
| 15   | 2023 | ICCV    | FerKD: Surgical Label Adaptation for Efficient Distillation  | <li> 针对蒸馏训练教师模型的额外开销问题，FKD的问题<br><li> 提出自适应标签校正，丢弃非常简单的和非常困难的样本，重点优化剩余样本<br><li> 提出一种新的增强方法 | 蒸馏训练优化<br/>教师推理开销        |
| 16   | 2023 | ICCV    | ORC: Network Group-based Knowledge Distillation using Online Role Change | <li> 提出分组教学框架处理教师将错误知识传递给学生的问题<br><li> 提出强化教学针对学生觉得困难的样本<br><li>提出私人教学更新学生模型中最好的模型<br><li> 提出组教学让教师和最好的学生教授剩余模型 | 蒸馏训练优化<br>减少错误传递         |
|      | 2023 | arXiv   | Norm KD: Normalized Logits for Knowledge Distillation        | <li> 认为固定单一的温度不足以软化不同的样本<br><li> 提出将样本输出logits视为高斯分布，将其标准化后的结果相当于利用其标准差作为温度系数<br><li>在标准化温度系数中引入一个缩放因子来调整分布 | 温度系数                             |
|      | 2024 | arXiv   | Dynamic Temperature Knowledge Distillation                   | <li>提出使用logsumexp函数来评价logits的平滑度，通过调节温度系数使得师生模型logit之间的平滑度差异最小<br><li>设定一个相同的初始温度，尖锐度较大的教师温度设为$\tau+\delta$，尖锐度较小的教师温度设为$\tau-\delta$<br><li>能够很好的与其他方法结合 | 温度系数<br>师生差距<br>任务难度调整 |
|      | 2024 | CVPR    | Logit Standardization in Knowledge Distillation              | <li> 从信息论的最大熵理论利用拉格朗日乘子法推导的温度系数<br><li>传统KD会迫使学生模型模仿教师模型的偏移量标准差<br><li> 提出利用样本均值标准差进行标准化来发挥温度系数的作用 | 温度系数<br>信息论角度推导           |

## 大模型蒸馏

| 序号 | 年份 | 出处    | 题目                                                         | 关键点                                                       | 目标                              |
| ---- | ---- | ------- | ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------- |
| 1    | 2021 | NeurIPS | Train Data-Efficient Image Transformers & Distillation Through Attention (DeiT) | <li> 用卷积网络指导ViT训练<br><li> 提出蒸馏token接收教师知识<br><li> 提出一种新的硬蒸馏策略 | token蒸馏<br>卷积老师             |
| 2    | 2022 | CVPR    | Co-advise: Cross Inductive Bias Distillation (CiT)           | <li> 对DeiT的改进 <br><li> 分析了不同教师归纳偏差的重要性<br><li> 提出多个token并用不同类型的教师教授ViT | token蒸馏<br>卷积老师<br>内卷老师 |
| 3    | 2023 | CVPR    | TinyMIM: An Empirical Study of Distilling MIM Pre-trained Models | <li> 研究了如何将MIM知识转移到较小网络中<br><li> 研究了蒸馏特征图 cls token 和 QKV关系的性能，蒸馏QKV最好<br><li> 提出类似助教的顺序蒸馏策略 | QKV蒸馏<br>MIM老师                |

