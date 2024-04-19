# Jensen's Inequality



## Convex Function 

>  特别指出，凸函数定义在国内某些教材上与国际上相反，以国际标准为准。

凸函数指函数图像上，任意两点连成的线段，皆位于图形上方的实值函数。形式化定义：

实值函数$f:C \rightarrow \R, \forall t\in[0, 1], \forall v, w\in C$:
$$
f[v + t·(w-v)] \leq f(v) - t·[f(w) - f(v)] \tag{1}
$$
 则$f$称为凸函数



## Jensen's Inequality

假设$f:I \rightarrow \R$为凸函数，则$\forall x_1, ..., x_n\in I, \forall t_1, ..., t_n\in[0, 1], t_1+t_2+...+t_n=1$, 有：
$$
f(t_1x_1+...+t_nx_n) \leq t_1f(x_1)+...+t_nf(x_n) \tag{2}
$$

#### Proof.

由数学归纳法：
$$
\begin{flalign}
& n=1: f(t_1x_1) \leq t_1f(x_1) \tag{3} &
\end{flalign}
$$

$$
\begin{flalign}
& n=2:f(t_1x_1+t_2x_2) \leq t_1f(x_1) + t_2f(x_2) \tag{4} &
\end{flalign}
$$

由凸函数定义上式显然成立。

假设：$n=k: \leq t_1f(x_1)+t_2f(x_2)+...+t_kf(x_k)$成立:
$$
\begin{flalign}
& n = k+1: &\\
& f(t_1x_1 + t_2x_2+...+ t_kx_x + t_{k+1}x_{k+1}) = f[(1-t_{k+1})\frac{t_1x_1 + t_2x_2+...+t_kx_x}{(1-t_{k+1})} + t_{k+1}x_{k+1}]
\end{flalign} \tag{5}
$$
由Eq(3), Eq(4):
$$
\begin{flalign}
& f[(1-t_{k+1})\frac{t_1x_1 + t_2x_2+...+t_kx_x}{(1-t_{k+1})} + t_{k+1}x_{k+1}]  &\\
& \leq (1-t_{k+1}) f(\frac{t_1}{1-t_{k+1}}x_1 + \frac{t_2}{1-t_{k+1}}x_2 + ... + \frac{t_k}{1-t_{k+1}}x_k) + t_{k+1}x_{k+1} \\
& \leq t_1f(x_1) + t_2f(x_2) + ... + t_kf(x_k) + t_{k+1}x_{x+1}
\end{flalign} \tag{6}
$$
得证。



### Probability Theory Version

对随机变量$X$, $\varphi$为凸函数：
$$
\varphi(E(X)) \leq E(\varphi(X)) \tag{7}
$$
即：

对$\int^\infty_{-\infty}f(x)dx = 1$，$\varphi$在$g$的值域上为凸函数：
$$
\varphi(\int^\infty_{-\infty}g(x)f(x)dx) \leq \int^\infty_{-\infty}\varphi(g(x))f(x)dx \tag{8}
$$
对$\Omega = \{x_1, x_2, ..., x_n\}, \sum_{i=1}^n\lambda_i=1, \lambda_i \geq0$：
$$
\varphi(\sum_{i=1}^ng(x_i)\lambda_i) \leq \sum^n_{i=1}\varphi(g(x_i))\lambda_i \tag{9}
$$

## 证明KL散度非负

对离散随机变量$\xi$，存在两个概率分布$P,Q$，则KL散度定义为：
$$
KL(P||Q) = \sum_iP(i)log\frac{P(i)}{Q(i)} \\
\sum_iP(i) = \sum_iQ(i) = 1 \tag{10}
$$

#### Proof.

$$
\sum_iP(i)log\frac{P(i)}{Q(i)} = \sum_iP(i)[-log\frac{Q(i)}{P(i)}] \geq -log[\sum_iP(i)\frac{Q(i)}{P(i)}] = -log\sum_iQ(i)=0 \\
KL(P||Q) \geq 0

\tag{11}
$$

