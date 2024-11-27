

# $C^2$KD: Bridging the Modality Gap for Cross-Modal Knowledge Distillation

**[CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)	no code	CREMA-D AVE VGGSound CrisisMMD	20240903	**跨模态

*Fushuo Huo, Wenchao Xu, Jingcai Guo, Haozhao Wang, Song Guo*

现有的蒸馏方法对跨模态蒸馏效果不理想，本文认为是模态之间的不平衡性和软标签错位导致的性能下降，提出了**通过肯德尔相关系数选择相似性高的跨模态样本**，**利用代理模型进行双向蒸馏更新师生模型**，KRC控制双向蒸馏的通道，从而弥合跨模态之间的不平衡性和样本错位问题，提升跨模态蒸馏的性能。

## Introduction 

现有知识蒸馏方法在单一知识转移中取得了实质性的成功，但是现有方法很难扩展到跨模态知识蒸馏中，我们发现模态的不平衡性和软标签错位导致了传统KD方法在跨模态KD的失效。

我们将模态的不平衡定义为模态之间的性能差异，实验发现音频模态要优于视觉模态；

对于第二个因素，将软标签定义为教师网络的输出分布，但由于模态间的鸿沟导致师生模态之间严重的软标签错位，因此直接在模态之间传递软标签是不合理的，并通过坎德尔相关系数KRC量化了这种错位，多模态的KRC要显著低于单模态的KRC

为了解决这种问题，提出了通过On-the-fly Section Distillation(OFSD)来双向更新来自定义教师和学生模型，OFSD根据KRC有选择性的

## Method

### 跨模态KD效率分析

利用DKD对损失函数解耦，分解为目标类TCKD和非目标类NCKD，实验发现模态的不平衡性会导致音频模态TCKD性能大幅下降，而由于模态之间的不相似性，模态之间的非目标信息是相互矛盾的，也会带来性能的损失。

$$
KRC = \frac{2}{C(C-1)}\sum_{i<j}sign(f^i_T - f^j_T)sign(f^i_S - f^j_S)
$$
我们认为秩相关性失衡是造成跨模态KD的原因，因此我们过滤掉了KRC<0的多模态样本，实验证明这改善了跨模态蒸馏的性能。**并且通过试验证明了KRC的有效性。**

**因此提出使用坎德尔相关系数KRC来衡量秩相关性**：

### 自定义跨模态蒸馏

![image-20240903104924248](.\imgs\image-20240903104924248.png)

我们认为教师模态和学生模态之间是互补的，软标签中的错位样本应当被过滤掉，**提出在线选择蒸馏OFSD策略，以排除不可蒸馏的样本并从非目标类中继承知识**，样本选择策略：
$$
\eta = \begin{cases}
\begin{aligned}
1,\ &KRC(f_T, f_S) > \omega \\
0,\ &otherwise
\end{aligned}
\end{cases}
$$
$\eta \in [0, 1]$为OFSD滤波器，$\omega$为阈值参数。

我们还提出构建双代理来逐步产生软标签：
$$
f_m = f_{c_m}^{cls}(GAP(B_m(F_m))), m\in \{T, S\} \\
f_m^{pro} = f_{c_m}^{cls(pro)}(A[GAP(B_m(F_m))]), m\in \{T, S\}
$$
GAP表示全局平均池化，$f_{c}^{cls}$表示分类头，A表示特征适配层（Conv-BN-ReLU），利用学生和教师模型代理作为桥梁进行双向蒸馏，总体损失表示为：
$$
\begin{aligned}
L_{all} = 
	& H(\sigma(f_S), Y) + H(\sigma(f_T), Y) \\
	& + \lambda_1D(\sigma(f_T), \sigma(f_T^{pro})) + \lambda_1D(\sigma(f_T^{pro}), \sigma(f_T)) \\
	& + \lambda_2D(\sigma(f_S), \sigma(f_S^{pro})) + \lambda_2D(\sigma(f_S^{pro}), \sigma(f_S)) \\
	& + \lambda_3\eta D(\sigma(\hat{f}_S^{pro(i)}),\sigma(\hat{f}_T^{pro(i)})) + \lambda_3\eta D(\sigma(\hat{f}_T^{pro(i)}),\sigma(\hat{f}_S^{pro(i)}))
\end{aligned}
$$
$i \neq t$为非目标类样本，H D分别表示交叉熵和KL散度