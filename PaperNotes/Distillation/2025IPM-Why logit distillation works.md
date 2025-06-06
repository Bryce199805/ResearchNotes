# Why logit distillation works: A novel knowledge distillation technique by deriving target augmentation and logits distortion

2025 IPM	no code 	ImageNet1k CIFAR TinyImagenet STL10	20250509

## Introduction 

尽管logit蒸馏旨在将知识从大型教师网络传授给学生，但其有效性的潜在机制和原因尚不清楚，我们提出了两个问题：为什么老师的知识是有效的？如果该技术的秘密被揭露，是否有可能进一步改进这些方法？

正常知识是指教师模型为每个类分配的概率，以高置信度强调正确的类；暗知识捕获的教师模型类之间微妙的关系信息，反映在非目标类的较低置信度概率中，这些信息有助于学生模型学习复杂度类别关系，促进灵活性和更好的泛化。

正常知识会引导学生进行准确预测，但暗知识能使其更深入的理解类的相似性，从而减少过拟合并增强对数据变化的鲁棒性。本研究旨在提供可解释的KD来弥合其在理论上的缺失，这有助于开发完美的KD方法。我们的研究围绕三个问题：

- 类别之间的非目标关系是否会导致学生学习相同的类别模式？
- 暗知识是否为学生注入了随机性？
- 学生的表现是都因为监督灵活性的增强而得到改善？

我们的主要贡献：

- 阐明暗指是语义的概念，以更清楚的理解其在KD中的作用
- 通过注入随机性引入动态logit失真的新概念，并从预测和目标增强的角度检查KD的影响，提出一种新的蒸馏技术，利用噪声改进泛化
- 提供对使logit蒸馏成为一种有效技术机制的见解，并揭开其成功背后的原因



## Method

#### TALD-KD is a type of ensemble learning strategy

向教师logit中添加随机噪声类似于教师logit的混合，将随机噪声注入教师logits是一种避免过拟合的正则化技术，如果教师模型与训练数据集过拟合，则其logit可能包含高度特定的信息，从而导致学生模型记住样本而不是泛化。

添加噪声可以防止学生记住老师的输出，而是迫使他们学习更可推广的模式；TALD-KD促进稳定性，学生通过将学生模型暴露于教师logit的略微扰动版本，学会了对变化更加文案金。这样可以提高看不见的数据的性能；TALD-KD探索了假设空间，噪声注入有助于学生模型在训练期间探索更广泛的假设空间，这种探索可以防止学生模型收敛到对教师的精确logits过于特定的局部最小值，我们使用$z^T+\epsilon$而不是$z^T$，因此学生模型的训练目标表示为：
$$
L = E_{x,y\sim D}[L_{CE}(y, f^S(x)) + \lambda L_{KD}(f^S(x), z^T+\epsilon)]
$$

#### Theoretical proof

本研究从集成学习和贝叶斯学习理论中汲取概念，以证明TALD-KD如何提高泛化能力，多个模型组合在一起，以提高集成学习的整体性能，其核心思想是聚合多个模型的预测来减少方差和偏差，从而获得更好的泛化，具有M个教师的教师模型系统，集成的logit可以表示为：
$$
z^{ensemble} = \frac{1}{M}\sum^M_{i=1}z^T_i
$$
学生模型经过训练，可以最大限度的减少其预测与集成logits之间的损失，可以表述为：
$$
L = E_{x,y\sim D}[L_{CE}(y, f^S(x)) + \lambda L_{KD}(f^S(x), z^{ensemble})]
$$
学生收与使用集成logits的约化方差，单个教师模型可能会进行高方差预测，平均他们的logits可以减少防擦好，其次，TALD-KD减少了偏差，不同模型捕获了数据分布的不同方面。

从贝叶斯的角度来看，教师模型的集成可以看作是模型参数上贝叶斯后验分布的近似值，每个教师模型都表示有关数据分布的不同假设，在贝叶斯学习中，目标是在给定数据D的情况下，找到模型参数$\theta$上的后验分布$p(\theta|D)$，后验分布捕获的不是单个点的估计，而是$\theta$的不确定性

设$z^{T, \theta}$表示由$\theta$参数化的教师模型的logits，集成方法近似于对后验分布的期望：
$$
E_{\theta\sim p(\theta|D)}[z^{T, \theta}] \approx \frac{1}{M}\sum^M_{i=1}z^{T, \theta_i}
$$
$\theta_i$表示从后验分布$p(\theta|D)$中采样的结果，训练学生模型以匹配该期望值，对应于最小化损失：
$$
L = E_{x, y\sim D}[L_{CE}(y, f^S(x)) + \lambda L_{KD}(f^S(x), E_{\theta\sim p(\theta|D)}[z^{T, \theta}])]
$$
通过考虑多个教师模型，学生可以更好的捕捉预测的不确定性，实现更稳健的泛化；对后验分布的期望是一个正则化器，可以防止学生过度拟合任何老师的预测。

将噪声注入教师logit是隐式集成学习的一种形式，当随机噪声$\epsilon$添加到单个教师模型logit中时：
$$
z^T_{noisy} = z^T + \epsilon
$$
这可以解释为从$z^T$周围的教师模型分布中抽样，这种噪声注入模拟里具有略微不同的教师模型集合的效果，通过类似的机制促进泛化，通过随机性和稳健性来防止过拟合的正则化，添加噪声可以减少方差，捕获更全面的数据分布，并防止过度拟合，从而在学生模型中实现更好的泛化了。

