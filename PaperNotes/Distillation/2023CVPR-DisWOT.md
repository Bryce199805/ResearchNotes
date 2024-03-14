# DisWOT: Student Architecture Search for Distillation Without Training

CVPR2023

- 与基于训练的学生架构搜索需要单独训练或权重共享训练不同，我们的方法在搜索阶段不需要训练学生模型，且我们初始化使用小批量数据，计算效率高。
- 我们的方法是一种教师感知的蒸馏搜索，比传统的NAS有更好的预测蒸馏精度
- 我们的方法利用神经网络之间的高阶知识距离，来连接知识蒸馏和0代理的NAS

相关工作中总结了之前的自适应KD方法，读一读。



## Methodology

### 最优学生网络搜索

首先提出一个学生网络评价指标

**Semantic Similarity Metric:** 

### 高阶知识蒸馏

