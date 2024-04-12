# Training Deep Neural Networks in Generations: A More Tolerant Teacher Educates Better Students

**[AAAI 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4506)	no code 	CIFAR100  ILSVRC2012	20240410**

*Chenglin Yang  Lingxi Xie  Siyuan Qiao  Alan L. Yuille*

这项工作认为教师模型使用硬标签训练过于严格，次要类中也包含这许多信息，应该训练一个更宽容的教师来教授学生，这项工作提出要考虑次要类别的影响，视觉上相似的物体同样能够提供有用的信息，在训练教师模型的损失中添加一项，计算主类与其他k-1个类别之间的置信度差异，通过这种手段来加强次要类别的置信度从而训练出一个更宽容的教师。

## Introduction

现有方法大多使用硬标签来训练教师网络，这导致一个严格的教师本身具有很高的准确性，但我们认为教师需要更宽容，尽管这意味着更低的准确率。除了最大化主类的置信度外，允许保留一些次要类（视觉上与gt相似的东西）可能有助于缓解过拟合的风险。

我们在训练教师网络的标准交叉熵损失中增加了一个额外项，使其能够将置信度分配给少数几个次要类别，尽管这损害了教师网络的准确性，但他确实为学生网络提供了更大的空间。其结果表明要比使用硬标签学习的教师有着更好的指导效果。

## Method

### Teacher-Student Optimization

考虑一个标准的网络优化任务，给定一个参数化形式$y=f(x;\theta)$模型M，x,y为输入和输出，$\theta$为可学习参数，给定一个训练集$\mathcal{D}=\{ (x_1, y_1), ..., (x_N, y_N) \}$，目标是确定最适合这些数据的参数$\theta$。常规方法是将权重设为随机值，通过梯度下降进行逐步更新， 每次从数据集D中采样一个子集B：
$$
\mathcal{L}(\mathcal{B}, \theta) = \frac{1}{|\mathcal{B}|}\sum_{(x_n, y_n)\in \mathcal{B}}y^T_n\ ln\ f(x_n;\theta)
$$
然而这种方式很容易过拟合，软化监督信号的一种方式是进行师生优化，首先使用上式训练一个$\mathbb{M}^T:f(x;\theta^T)$，然后使用混合损失训练一个学生模型$\mathbb{M}^S:f(x;\theta^S)$:
$$
\mathcal{L}^S(\mathcal{B}, \theta^S) = \frac{1}{|\mathcal{B}|}\sum_{(x_n, y_n)\in \mathcal{B}}\{ \lambda·y_n^T\ ln f(x_n;\theta^S) + (1-\lambda)·KL[f(x_n;\theta^T) || f(x_n;\theta^S)]\}
$$
一个直接的扩展是允许一个网络在多代中被优化，训练一个父辈模型记为$\mathbb{M}^0$，只受数据集的监督。第m代在教师$\mathbb{M}^{m-1}$的监督下训练出$\mathbb{M}^m$，我们**观察到在前几个，识别准确率上升，但后来开始饱和并下降**，我们将分析这种现象的原因。原因是几代之后主类的值趋向于1，教师由转变为之前的严格教师，性能开始饱和甚至下降。

### Preserving Secondary Information: An Important Factor in Teacher-Student Optimization

先前工作都专注于从较大的网络中提取知识，而忽视了类别之间的相似性。我们研究了BANS(2018PMLR)，由软标签的指导，学生模型获得了更好的性能。因此我们提出一个问题：**通过软标签进行训练的关键好处是什么？**

我们研究发现，一个深度神经网络能够自动的为每张图像单独学习语义相似的类，我们将其命名为次要信息。利用这些信息学生模型可以避免拟合不必要的严格分布，从而更好的泛化。

### Towards High-Quality Secondary Information

我们注意到次要信息的关键是软化输出特征向量，我们考虑三种途径：前两种方法遵循先前的工作，标签平滑正则化LSP和置信度惩罚CP，其均在原来的交叉熵损失的基础上增加一个额外id像，将softmax之后的分数分布优化到峰值更小的分布。LSR增加的是输出分布与均匀分布之间的KL散度，CP中是一种负熵增益。他们有个共同的缺点，他们可以使置信度分布在所有类中而不管这些类是否与训练样本在视觉上相似。

我们提出更合理的方法，我们挑选几个被分配了最高置信度分数的类，并假设这些类可能与输入图像在语义上相似，我们设置一个固定的整数K，他表示每幅图像包含的语义合理的类被书，其中包括主类别，我们计算主类别和其他K-1个类别之间置信度的差距：
$$
\mathcal{L}^T(\mathcal{B}, \theta^T) = \frac{1}{|\mathcal{B}|}\sum_{(x_n, y_n)\in \mathcal{B}}\{ -\eta·y^T_nln\ f(x_n;\theta^T) + (1-\eta)·[f_{a_1} - \frac{1}{K-1}\sum^K_{k=2}f_{a_k}] \}
$$
我们称之为最高得分差TSD，$f^T_{a_k}$为$f(x_n;\theta^T)$输出中第k大的元素。
