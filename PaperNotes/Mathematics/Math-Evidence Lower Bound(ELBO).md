# Evidence Lower Bound

在变分贝叶斯方法中，变分下界(Evidence Lower Bound, ELBO)是一种用于估计一些观测数据的对数似然下界。

## Definition

设X和Z为随机变量，其联合分布为$p_\theta$，例如$p_\theta(X)$是X的边缘分布，$p_\theta(Z|X)$为在给定X的条件下Z的条件分布，则对任何从$p_\theta$中抽取的样本$x \sim p_\theta$和任何分布$q_\phi$，有：
$$
log\ p_\theta(x) \geq \mathbb{E}_{z\sim q_\phi}[ln\frac{p_\theta(x, z)}{q_\phi(z)}]
$$
上式称为ELBO不等式，左侧为x的证据，右侧为证据下界。

## Motivation

假设我们有一个可观察的随机变量X，且我们想找其真实分布$p^*$，这将允许我们抽样生成数据来估计未来事件概率。但是精确找到$p^*$是不可能的，需要寻找一个近似。定义一个足够大的参数化分布族$\{ p_\theta \}_{\theta\in \Theta}$， 最小化某种损失函数L，$\underset{\theta}{min\ L(p_\theta, p^*)}$。解决该问题的一种方法是考虑从$p_{\theta}$到$p_{\theta+\delta\theta}$的微小变化，并使得$L(p_\theta, p^*) - L(p_{\theta+\delta\theta}, p^*) = 0$，这是变分法中的一个变分问题。

我们考虑隐式参数化的概率分布：

- 定义一个在潜在随机变量Z上的简单分布p(z) （高斯分布、均匀分布等）
- 定义一个由$\theta$参数化的复杂函数族$f_\theta$（神经网络）
- 定义一种将任何$f_\theta(z)$转化为可观测随机变量X的简单分布的方法，如$f_{\theta}(z) = (f_1(z), f_2(z))$，可以将相应分布X定义在高斯分布上$N(f_1(z), e^{f_2(x)})$

这构造了一个关于（X， Z）的联合分布族$p_\theta$， 从$p_\theta$中抽取样本$(x, z) \sim p_\theta$非常容易，秩序从p中抽样z~p, 然后计算$p_\theta(z)$，最后使用$f_\theta(z)$来抽样$x\sim p_\theta(·|z)$。我们拥有了一个可观测和潜在随机变量的生成模型。

我们想要构造一个$p^*$使得$p^*(X) \approx p_\theta(X)$， 右侧需要对Z进行边缘化来消除Z的影响。我们无法计算$p_\theta(x) = \int p_\theta(x|z)p(z)dz$需要寻找一个近似。根据贝叶斯公式：
$$
p_\theta(x) = \frac{p_\theta(x|z)p(z)}{p_\theta(z|x)}
$$
我们需要找到一个$q_\phi(z)$来近似$p_\theta(z|x)$，这是一个针对潜变量的判别模型

##  Deriving the ELBO

引入一个新的分布$q_\phi(z)$作为潜变量z的后验分布$p_\theta(z|x)$的近似，边际对数似然$log\ p_\theta(x)$可以表示为：  
$$  
\begin{aligned}
log\ p_\theta(x) &= log[\frac{p_\theta(x, z)}{p_\theta(z|x)}] = log\ p_\theta(x, z) - log\ p_\theta(z|x) \\
&= log\ p_\theta(x, z) -log\ q_\phi(z)  - log\ p_\theta(z|x) + log\ q_\phi(z) \\
&= log[\frac{p_\theta(x, z)}{q_\phi(z)}] + log[\frac{q_\phi(z)}{p_\theta(z|x)}]
\end{aligned}
$$

两边求$q_{\phi}(z)$的期望：

$$
\begin{aligned}
&\mathbb{E}_{z\sim q_\phi(z)}[log\ p_\theta(x)] = \sum_z q_\phi(z) log\ p_\theta(x) = log\ p_\theta(x)\sum_zq_\phi(z) = log\ p_\theta(x) \\
&\mathbb{E}_{z\sim q_\phi(z)}[log[\frac{p_\theta(x, z)}{q_\phi(z)}] + log[\frac{q_\phi(z)}{p_\theta(z|x)}]] = \mathbb{E}_{z\sim q_\phi(z)}[log[\frac{p_\theta(x, z)}{q_\phi(z)}]] + \mathbb{E}_{z\sim q_\phi(z)}[log[\frac{q_\phi(z)}{p_\theta(z|x)}]] \\
&=\sum_z q_\phi(z) log[\frac{p_\theta(x, z)}{q_\phi(z)}] + \sum_zq_\phi(z)log[\frac{q_\phi(z)}{p_\theta(z|x)}] = \mathcal{L} + KL(q_\phi(z) || p_\theta(z|x)) \\
\end{aligned}
$$
因此有：
$$
log\ p_\theta(x) = \mathcal{L} + KL(q_\phi(z) || p_\theta(z|x)) \\
\mathcal{L}_{ELBO} = \sum_z q_\phi(z) log[\frac{p_\theta(x, z)}{q_\phi(z)}] = \mathbb{E}_{z\sim q_\phi(z)}[log[\frac{p_\theta(x, z)}{q_\phi(z)}] = log\ p_\theta(x) - KL(q_\phi(z) || p_\theta(z|x))
$$
由此推出ELBO变分下界，其中KL散度是非负的，$\mathcal{L}_{ELBO} \leq log\ p_\theta(x)$，因此可以通过最大化ELBO来近似最大化边际对数似然。

## Reference

https://zh.wikipedia.org/wiki/%E8%AF%81%E6%8D%AE%E4%B8%8B%E7%95%8C

https://0809zheng.github.io/2020/03/25/variational-inference.html