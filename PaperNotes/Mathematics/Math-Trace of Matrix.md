# 矩阵的迹

## 迹的性质

$$
tr(a) = a
$$

$$
tr(A) = tr(A^T) 
$$

$$
tr(A+B) = tr(A) + tr(B)
$$

$$
tr(AB) = tr(BA)
$$

$$
tr(ABC) = tr(BCA) = tr(CAB)
$$

$$
tr(A) = \sum^n_{k=1}\lambda_k
$$

$$
tr(x^TAx) = tr(Axx^T) = x^TAx
$$

## 求导法则

$$
\frac{\partial tr(X)}{\partial X} = I
$$

$$
\frac{\partial tr(g(X))}{\partial X} = g'(X)
$$

$$
\frac{\partial tr(AX)}{\partial X} = \frac{\partial tr(XA)}{\partial X} = A^T
$$

$$
\frac{\partial tr(AX^T)}{\partial X} = \frac{\partial tr(X^TA)}{\partial X} = A
$$

$$
\frac{\partial tr(AXB)}{\partial X} = \frac{\partial tr(XBA)}{\partial X} = (BA)^T = A^TB^T
$$

$$
\frac{\partial tr(AXBX^T)}{\partial X} = AXB + A^TXB^T
$$

