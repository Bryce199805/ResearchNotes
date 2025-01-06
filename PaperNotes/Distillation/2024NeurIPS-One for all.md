# One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation

[2024 NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html)



## Introduction

现有基于中间层的蒸馏方法主要集中在同质架构之间的蒸馏，而对于跨架构模型的蒸馏没有得到探索，在实际应用中并不总是能够找到与学生架构相匹配的同质模型。

异构模型的情况下不能保证学习到的特征能够对齐，直接匹配这些信息意义不大，甚至会阻碍学生模型的性能。

因此本文提出一个one-for-all 的知识蒸馏框架，用于异构模型之间的蒸馏。

## Method

#### 针对异构架构的特征蒸馏

