# Kernel Method

**核方法**是指通过**核技巧**把样本输入空间的非线性问题转换为特征空间的线性问题，通过引入**核函数**不显式地定义特征空间和映射函数，而是将优化问题的解表示成核函数的**线性组合**。

对于一个线性模型$f(x)=w^Tx$忽略偏置项b，根据表示定理，在满足一定条件时，参数w的最优解可以表示为N个样本数据的线性组合$w^*=\sum^N_{i=1}\alpha_ix_i$，因此最优解表示为：
$$
f^*(x)=\sum^N_{i=1}\alpha_ix_i^Tx
$$
引入映射$\phi:\mathcal{X} \rightarrow \mathcal{F}$，将样本空间$\mathcal{X}$变换到高维的特征空间$\mathcal{F}$，从而将在样本空间中线性不可分的样本在特征空间中线性可分（**若原始样本空间为有限维，则一定存在一个高维的线性空间使得样本线性可分**），为模型增加非线性的表示能力，则对应模型最优解为：
$$
f^*(x)=\sum^N_{i=1}\alpha_i\phi(x_i)^T\phi(x)
$$
直接计算特征变换$\phi(x)$以及特征变换的内积$\phi(x)^T\phi(x')$相当困难，这可能会把样本投射到无穷维度的特征空间，最优解中仅出现了特征变换的内积，因此引入核函数$k(x, x')=\phi(x)^T\phi(x')$，将计算特征变换的内积$\phi(x)^T\phi(x')$转化为计算核函数的值$k(x, x')$。

- **核技巧(Kernel Trick)**是指通过一个非线性变换$\phi$把输入空间(如低维的欧式空间)映射到特征空间(如高维的希尔伯特空间)，从而把输入空间的非线性问题转换为特征空间的线性问题
- **核函数(Kernel Function)**是指引入函数$k(x, x')$用来替换特征变换的内积$\phi(x)^T\phi(x')$计算。
- **核方法(Kernel Method)**是指满足表示定理的模型的目标函数和最优解只涉及输入之间的内积，通过引入核函数不需要显式地定义特征空间和映射函数，可以直接获得结果。



## 表示定理 Representer Theorem

我们重点关注线性模型的表示定理，表示定理适合用于任何模型h，只要该模型的优化问题可以构成如下结构风险与经验风险之和：
$$
\underset{h\in \mathcal{H}}{min} \ \Omega(||h||_\mathcal{H}) + \frac{1}{N}\sum^N_{i=1}l(y_i, h(x_i))
$$
其中$\mathcal{H}$为核函数k对应的再生希尔伯特空间，$||h||_{\mathcal{H}}$表示$\mathcal{H}$空间中关于h的范数，要求$\Omega$是单调递增的函数，$l$是非负损失函数，上述优化问题的最优解可以表示为核函数的线性组合：
$$
h^*(x)=\sum^N_{i=1}\alpha_ik(x_i, x)
$$

### 一种特殊形式下的证明：线性模型+L2正则化

线性模型$f(x)=w^Tx$使用了L2正则化，优化目标函数：
$$
\underset{w}{min}\  \frac{\lambda}{N}w^Tw + \frac{1}{N}\sum^N_{i=1}l(y_i, w^Tx_i)
$$
参数w的最优解表示为所有样本的线性组合：
$$
w^* = \sum^N_{i=1}\alpha_ix_i
$$

#### Proof:

最优解w\*与样本x具有相同的空间维度，假设最优解由两部分组成：$w^*=w_{||}+w_{\perp}$，$w_{||}$表示平行于样本数据所构成的空间$span(x_1,...,x_N)$；$w_{\perp}$垂直与样本数据构成的空间。

因此：
$$
l(y_i, w^{*T}x_i) = l(y_i, (w_{||}+w_{\perp})^Tx_i) = l(y_i, w_{||}^Tx_i) \\
w^{*T}w^* =  (w_{||}+w_{\perp})^T (w_{||}+w_{\perp})=w_{||}^Tw_{||} + 2w_{||}^Tw_{\perp} + w_{\perp}^Tw_{\perp} = w_{||}^Tw_{||} + w_{\perp}^Tw_{\perp} \geq w_{||}^Tw_{||}
$$
显然参数的一个可选解$w_{||}$比假设最优解w*具有更小的目标函数，因此$w_{||}$是一个满足条件的更优解，最优参数w\*平行于样本数据构成的空间$span(x_1,...,x_N)$，即可被样本数据线性表示

##  核函数的定义

### Definition 1

对于函数$k:\mathcal{X} \times \mathcal{X} \rightarrow \R$，如果存在映射$\phi:\mathcal{X} \rightarrow \R, \phi\in \mathcal{H}$，使得：
$$
k(x, x')=<\phi(x), \phi(x')>
$$
则称$k(x, x')$为正定核函数。其中$\mathcal{H}$是希尔伯特空间(Hilbert Space)，即完备的、可能是无限维的、被赋予内积的线性空间。

- 完备：对极限是封闭的：$\forall\ Cauchy Sequence:\{x_n\}, \underset{n \rightarrow \infty}{lim} x_n=x\in \mathcal{H}$
- 内积：满足线性、对称性和非负性的内积运算
- 线性空间：对加法和数乘封闭的向量空间



### Definition 2

对于函数$k:\mathcal{X}\times \mathcal{X} \rightarrow \R$，如果满足下面两条性质：

- $k(x, x')$是**对称函数**，即$k(x, x') = k(x', x)$
- 对任意样本集$x=\{x_1, x_2,...,x_N \}^T$，其中Gram矩阵（核矩阵）$K=[k(x_i, x_j)]_{N\times N}$是**半正定矩阵**

则称$k(x, x')$为正定核函数。即一个**对称函数所对应的核矩阵半正定，该函数就能够作为核函数**，所以称为正定核函数。



### 两种定义的等价性

#### Proof:

对称性： $k(x, x') = <\phi(x), \phi(x')> = <\phi(x'), \phi(x)> = k(x', x)$

正定性：由正定定义，引入$\alpha\in \R^N$:
$$
\alpha^TK\alpha = [\alpha_1, \alpha_2, ..., \alpha_N][k(x_i, x_j)]_{N\times N}[\alpha_1, \alpha_2, ..., \alpha_N]^T \\
=\sum^N_{i=1}\sum^{N}_{j=1}\alpha_i\alpha_jk(x_i, x_j)=\sum^N_{i=1}\sum^{N}_{j=1}\alpha_i\alpha_j<\phi(x_i), \phi(x_j)> \\
=\sum^N_{i=1}\sum^{N}_{j=1}\alpha_i\alpha_j\phi(x_i)^T\phi(x_j)=\sum^N_{i=1}\alpha_i\phi(x_i)^T\sum^{N}_{j=1}\alpha_j\phi(x_j) \\
=(\sum^N_{i=1}\alpha_i\phi(x_i))^T (\sum^{N}_{j=1}\alpha_j\phi(x_j))=<\sum^N_{i=1}\alpha_i\phi(x_i), \sum^{N}_{j=1}\alpha_j\phi(x_j)> = ||(\sum^N_{i=1}\alpha_i\phi(x_i))||^2 >0
$$

## 常用核函数

对于一个核函数，总能找到一个对应的映射$\phi$，即任何核函数都隐式的顶一个了一个称为**再生核希尔伯特空间**的特征空间，通常希望样本在特征空间中是线性可分的，选择核函数相当于选择特征空间，因此选择合适的核函数是核方法中的重要问题。

### 线性核(Linear Kernel)

线性核函数定义为：
$$
k(x,x')=x^Tx'
$$

- 优点：模型简单，速度快，可解释性好
- 缺点：无法处理线性不可分的数据集

### 二项式核(Binomial Kernel)

对于输入样本$x=(x_1, x_2,..., x_d)^T$，定义二阶多项式变换得到的特征向量：
$$
z=\psi(x)=(1, x_1,...,x_d, x_1^2, x_1x_2,...,x_d^2)^T
$$
由核函数定义：
$$
k(x, x')=\varphi(x)^T\varphi(x') = 1+\sum^d_{i=1}x_ix_i'+\sum^d_{i=1}\sum^d_{j=1}x_ix_i'x_jx_j'=1+x^Tx'+(x^Tx')^2
$$
对特征变换进行一些修改，可以得到对应的核函数：
$$
\varphi(x) = (1, \sqrt2x_1,...,\sqrt2x_d,x_1^2,x_1x_2,...,x_d^2)^T \rightarrow k(x, x')=(1+x^Tx')^2 \\
\varphi(x) = (1, \sqrt{2\gamma}x_1,...,\sqrt{2\gamma}x_d,\gamma x_1^2,\gamma x_1x_2,...,\gamma x_d^2)^T \rightarrow k(x, x')=(1+\gamma x^Tx')^2 \\
\varphi(x) = (\zeta, \sqrt{2\zeta\gamma}x_1,...,\sqrt{2\zeta\gamma}x_d,\gamma x_1^2,\gamma x_1x_2,...,\gamma x_d^2)^T \rightarrow k(x, x')=(\zeta+\gamma x^Tx')^2
$$

### 多项式核(Polynomial Kernel)

扩展二项式核得到多项式核：
$$
k(x,x')=(\zeta+\gamma x^Tx')^q,\ \zeta \geq0,\gamma>0
$$

- 优点：模型非线性；可控制阶数q
- 缺点：由于幂运算，数值稳定性差，需要选择三个超参数

### 高斯核(Gaussian Kernel)

高斯核函数又称为径向基函数(Radial Basis Function, RBF)，是一种无限维度的特征变换，高斯核函数为：
$$
k(x, x')=exp(-\gamma||x-x'||^2)
$$
其中超参数$\gamma>0$，控制高斯函数的方差，$\gamma$取值越大，径向基函数径向越小，模型越容易过拟合。当超参数$\gamma$趋向于无穷大时，高斯核趋近于：
$$
k(x, x')=
\begin{cases}
0,\  x\neq x' \\
1,\  x=x'
\end{cases}
$$

- 优点：模型更强大，数值稳定性好，只需要选择一个超参数
- 缺点：可解释性差，计算速度慢，容易过拟合

令$\gamma=1$，分析其对应的特征变换：
$$
k(x,x')=exp(-||x-x'||^2)=exp(-x^2)exp(-x'^2)exp(2xx') \\
= exp(-x^2)exp(-x'^2)\sum^\infty_{n=0}\frac{(2xx')^n}{n!} = \sum^\infty_{n=0}exp(-x^2)exp(-x'^2)\sqrt{\frac{2^n}{n!}}\sqrt{\frac{2^n}{n!}}x^nx'^n\\
=\sum^{\infty}_{n=0}(exp(-x^2)\sqrt{\frac{2^n}{n!}}x^n)(exp(-x^2)\sqrt{\frac{2^n}{n!}}x^n) = \varphi(x)^T\varphi(x')
$$
可得到特征变换$\varphi(x)$:
$$
z = \varphi(x) = exp(-x^2)(1, \sqrt{\frac{2^1}{1!}}x^1, \sqrt{\frac{2^2}{2!}}x^2, ..., \sqrt{\frac{2^n}{n!}}x^n)^T
$$

### 拉普拉斯核(Laplacian Kernel)

$$
k(x, x') = exp(-\frac{||x-x'||}{\sigma}), \sigma>0
$$

### Sigmoid核

$$
k(x, x') = tanh(\beta x^Tx'+\theta), \beta >0,\theta<0
$$

### 核函数的线性组合

若$k_1, k_2$为核函数，则对任意正数$\gamma_1, \gamma_2$，其线性组合仍然为核函数：$k = \gamma_1k_1+\gamma_2k_2$

若$k_1, k_2$为核函数，则核函数的直积仍为核函数：$k = k_1 \otimes k_2(x,x') =  k_1(x,x')  k_2(x,x')$

**若$k_1$为核函数，对任意函数g(x)，下列函数仍为核函数：$k(x,x')=g(x)k_1(x, x')g(x')$**



**[reference](https://0809zheng.github.io/2021/07/23/kernel.html)**
