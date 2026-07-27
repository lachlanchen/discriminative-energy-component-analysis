# DECA 理论与新颖性规格

> 工作名：Discriminative Energy Component Analysis（DECA）
> 建议论文副标题：Optimal Commuting Measurements for Structured Quadratic Classification
> 状态：算法、证明与已执行实验的冻结规格
> 日期：2026-07-27

## 1. 研究问题

设样本 \(x\) 经编码后成为单位向量

\[
\phi(x)\in\mathbb C^d,\qquad \|\phi(x)\|_2=1,
\]

并写成纯态

\[
\rho_x=\phi(x)\phi(x)^\dagger.
\]

类别 \(c\in\{1,\ldots,K\}\) 的条件算符为

\[
\rho_c=\mathbb E[\rho_x\mid y=c],
\qquad
\rho_c\succeq0,
\qquad
\operatorname{Tr}(\rho_c)=1.
\]

令类别先验为 \(\pi_c\)，并定义加权类算符

\[
A_c=\pi_c\rho_c.
\]

我们希望学习一个测量，使真实类别对应的测量结果具有最大概率。核心问题是：

> 在只允许一次共享基变换、一次计算基测量和一个经典 outcome-to-class decoder 的限制下，怎样得到最优或接近最优的类别判别测量？

这个限制同时具有三种解释：

- 机器学习：共享本征基的结构化二次分类器；
- 统计学：比较类条件二阶能量；
- 量子实现：不使用 ancilla 的 projective measurement compilation。

## 2. 必须承认的已有工作

以下内容不是本工作的原创贡献：

1. Helstrom 二元最小错误量子态判别；
2. 用类内纯态平均值构成 quantum centroid；
3. 用 Helstrom measurement 做经典二分类；
4. 用 Pretty Good Measurement（PGM）做多分类；
5. 用半正定规划求一般多元最优 POVM；
6. 训练 quantum embedding 后使用 Helstrom measurement；
7. stereographic encoding 和 tensor-copy encoding。

直接相关文献包括：

- Sergioli, Giuntini, and Freytes, “A New Quantum Approach to Binary
  Classification,” PLOS ONE, 2019:
  [paper](https://doi.org/10.1371/journal.pone.0216224)。
- Giuntini et al., “Quantum State Discrimination for Supervised
  Classification,” 2021:
  [arXiv:2104.00971](https://arxiv.org/abs/2104.00971)。
- Lloyd et al., “Quantum Embeddings for Machine Learning,” 2020:
  [arXiv:2001.03622](https://arxiv.org/abs/2001.03622)。
- Eldar, Megretski, and Verghese, “Designing Optimal Quantum Detectors via
  Semidefinite Programming,” IEEE Transactions on Information Theory:
  [arXiv:quant-ph/0205178](https://arxiv.org/abs/quant-ph/0205178)。
- Hwang et al., “Quantum-inspired Classification via Efficient Simulation of
  Helstrom Measurement,” 2024:
  [arXiv:2403.15308](https://arxiv.org/abs/2403.15308)。

因此，DECA 的新颖性不能写成“首次使用 Helstrom/POVM/density matrix
做分类”。

## 3. 一般 POVM oracle

一般最小错误判别求解

\[
\boxed{
S_{\mathrm{POVM}}^\star
=
\max_{\{E_c\}}
\sum_{c=1}^K\operatorname{Tr}(A_cE_c)
}
\]

满足

\[
E_c\succeq0,
\qquad
\sum_{c=1}^KE_c=I.
\]

这是凸半正定规划。其对偶为

\[
\boxed{
S_{\mathrm{POVM}}^\star
=
\min_Y\operatorname{Tr}(Y)
\quad
\text{s.t.}\quad
Y\succeq A_c,\ \forall c.
}
\]

它给出：

- 可数值求得的全局最优成功率；
- 多分类 DECA 的 oracle upper bound；
- primal-dual certificate；
- 一个需要 Naimark dilation 才能通用实现的量子测量。

本工作使用这个已知 SDP 作为 oracle，而不把 SDP 本身声称为新贡献。

## 4. 为什么学习一个公共 unitary 对不受限 POVM 没有意义

令 \(U\) 为任意 unitary，并对所有类别作相同变换

\[
\widetilde A_c=UA_cU^\dagger.
\]

对任意可行 POVM \(\{E_c\}\)，令

\[
\widetilde E_c=UE_cU^\dagger.
\]

则

\[
\sum_c\operatorname{Tr}(\widetilde A_c\widetilde E_c)
=
\sum_c\operatorname{Tr}(A_cE_c).
\]

因此

\[
S_{\mathrm{POVM}}^\star(\{\widetilde A_c\})
=
S_{\mathrm{POVM}}^\star(\{A_c\}).
\]

### 结论

> 如果后面的测量完全不受限，学习一个对所有样本共同作用的正交或
> unitary 变换不能提高最优可分辨性；该变换可以被吸收到测量算符中。

这解释了 ECA 的真正角色：

> ECA 不是通过共同 unitary 创造新的信息，而是学习一个基，使受限、便宜、
> 可解释的测量尽量接近不受限 POVM oracle。

## 5. DECA：最优可交换测量

限制所有 measurement effects 共享一个正交/酉基

\[
P=[p_1,\ldots,p_d],
\qquad
P^\dagger P=I.
\]

一般 commuting POVM 可写成

\[
E_c
=
P\operatorname{diag}(\ell_{1c},\ldots,\ell_{dc})P^\dagger,
\]

其中

\[
\ell_{jc}\ge0,
\qquad
\sum_c\ell_{jc}=1.
\]

成功率为

\[
\begin{aligned}
S(P,L)
&=\sum_c\operatorname{Tr}(A_cE_c)\\
&=\sum_{j,c}\ell_{jc}\,p_j^\dagger A_cp_j.
\end{aligned}
\]

定义

\[
a_{jc}(P)=p_j^\dagger A_cp_j.
\]

## 6. 定理一：固定基时 soft mapping 必然有 hard optimum

对固定 \(P\)，每一行 \(\ell_{j:}\) 独立位于概率单纯形。由于目标关于
\(\ell_{j:}\) 是线性的，

\[
\max_{\ell_{j:}\in\Delta_K}
\sum_c\ell_{jc}a_{jc}
=
\max_ca_{jc}.
\]

因此存在最优解

\[
\ell_{jc}^\star
=
\mathbf 1\!\left[
c=\arg\max_r a_{jr}
\right].
\]

于是

\[
\boxed{
S_{\mathrm{DECA}}(P)
=
\sum_{j=1}^d
\max_c p_j^\dagger A_cp_j.
}
\]

### 含义

- 对 minimum-error objective，soft \(L\) 不是必要的；
- 每个测量 outcome 应由最大 posterior contribution 的类别赢得；
- 最优 commuting POVM 可选为 PVM 的 classical coarse-graining；
- 这正是“每个 eigencomponent 民主投票给最匹配类别”的严格版本。

如果存在 tie，可使用固定规则或按先验随机化。

## 7. 定理二：二分类具有闭式解并达到 Helstrom bound

令

\[
\Delta=A_1-A_2.
\]

对任意 \(P\)，

\[
\begin{aligned}
S_{\mathrm{DECA}}(P)
&=\sum_j\max
\left(
p_j^\dagger A_1p_j,
p_j^\dagger A_2p_j
\right)\\
&=
\frac12\sum_j
\left[
p_j^\dagger(A_1+A_2)p_j
+
\left|p_j^\dagger\Delta p_j\right|
\right]\\
&=
\frac12
\left[
1+
\left\|
\operatorname{diag}(P^\dagger\Delta P)
\right\|_1
\right].
\end{aligned}
\]

由 Schur--Horn 定理，

\[
\operatorname{diag}(P^\dagger\Delta P)
\prec
\lambda(\Delta).
\]

\(\ell_1\) 范数是 convex symmetric function，因此

\[
\left\|
\operatorname{diag}(P^\dagger\Delta P)
\right\|_1
\le
\|\lambda(\Delta)\|_1
=
\|\Delta\|_1.
\]

当 \(P\) 取为 \(\Delta\) 的本征基时等号成立。因此

\[
\boxed{
S_{\mathrm{DECA}}^\star
=
\frac12(1+\|\Delta\|_1)
=
S_{\mathrm{POVM}}^\star.
}
\]

最优 component assignment 为

\[
z_j=
\begin{cases}
1,&\lambda_j(\Delta)>0,\\
2,&\lambda_j(\Delta)<0.
\end{cases}
\]

零本征值方向不包含二元判别信息，可任意分配或作为 reject subspace。

### 贡献边界

这个闭式解本质上是 Helstrom measurement，不能作为新 Helstrom 理论。
新贡献在于：

- 从 ECA 的 component allocation 目标独立推导出该结构；
- 证明二分类 ECA 的严格最优形式；
- 明确指出原来的 gradient-based \(P,L\) 训练在这一目标下是不必要的；
- 为多分类受限测量算法提供可验证的 binary anchor。

## 8. 判别谱与低秩 component selection

对二分类，\(\Delta\) 的本征对

\[
\Delta p_j=\lambda_jp_j
\]

给出：

- \(\operatorname{sign}(\lambda_j)\)：component 支持的类别；
- \(|\lambda_j|\)：该 component 对类平均态可分辨性的贡献；
- \(\sum_j|\lambda_j|=\|\Delta\|_1\)：总判别能量。

定义累计判别能量

\[
R(r)
=
\frac{
\sum_{j=1}^r|\lambda|_{(j)}
}{
\sum_{j=1}^d|\lambda_j|
},
\]

其中按 \(|\lambda_j|\) 降序排列。可用 \(R(r)\ge\eta\) 选择低秩
discriminative components。

这比按总体方差选 PCA component 更直接对应类别差异。

### 8.1 单次 PVM 与判别谱不是同一个决策规则

Helstrom PVM 的两个 effects 为 \(\Delta\) 的正、负谱投影。若使用精确
outcome probabilities 后再作确定性决策，其分数差为

\[
g_{\mathrm{PVM}}(x)
=
\operatorname{Tr}(\operatorname{sign}(\Delta)\rho_x)
=
\sum_j\operatorname{sign}(\lambda_j)
|p_j^\dagger\phi(x)|^2.
\]

它只保留本征值的符号。这是单次测量错误率最优所需要的量，但并不保证
经典测试样本的 deterministic argmax accuracy 最优。

你的“最大类间差异”原始想法更直接对应差算符本身：

\[
\boxed{
g_{\mathrm{spec}}(x)
=
\operatorname{Tr}(\Delta\rho_x)
=
\sum_j\lambda_j|p_j^\dagger\phi(x)|^2.
}
\]

我们把这个规则称为 **Spectral-DECA**。它保留：

- \(\operatorname{sign}(\lambda_j)\)：类别方向；
- \(|\lambda_j|\)：类间差异大小；
- \(|p_j^\dagger\phi(x)|^2\)：测试样本在该方向上的能量。

PVM-DECA 与 Spectral-DECA 因而解决两个不同的操作任务：

| 模式 | component weight | 目标 | 量子推理 |
|---|---:|---|---|
| PVM-DECA | \(\operatorname{sign}(\lambda_j)\) | 单次测量成功率 | 一次测量得到类别 |
| Spectral-DECA | \(\lambda_j\) | 确定性二次判别 | 重复测量并估计加权期望 |

二分类 Spectral-DECA 不需要梯度训练；一次
\(\Delta=P\operatorname{diag}(\lambda)P^\dagger\) 的特征分解就是解析解。

若在 \(P^\dagger\) 后重复测量，令随机变量
\(\Lambda=\lambda_J\)，则

\[
\mathbb E[\Lambda\mid x]=g_{\mathrm{spec}}(x).
\]

由 Hoeffding 不等式，若
\(r_\lambda=\lambda_{\max}-\lambda_{\min}\)，用 \(N\) shots 的样本均值
\(\widehat g\) 满足

\[
\Pr(|\widehat g-g_{\mathrm{spec}}|\ge\epsilon)
\le
2\exp\!\left(-\frac{2N\epsilon^2}{r_\lambda^2}\right).
\]

因此它仍只需同一个无 ancilla 测量基，但以 shots 换取本征值幅度信息。

## 9. 定理三：多分类 commuting classes 时 DECA 全局最优

如果

\[
[A_c,A_r]=A_cA_r-A_rA_c=0
\qquad
\forall c,r,
\]

则这些 Hermitian operators 存在公共本征基 \(P\)。在该基中，

\[
A_c=P\operatorname{diag}(a_{1c},\ldots,a_{dc})P^\dagger.
\]

对任意 POVM，只需考虑各 \(E_c\) 在这个基中的对角元，因为
\(A_c\) 的非对角元为零。每个 outcome \(j\) 上的最优决策是

\[
\arg\max_c a_{jc}.
\]

因此

\[
\boxed{
[A_c,A_r]=0\ \forall c,r
\quad\Longrightarrow\quad
S_{\mathrm{DECA}}^\star
=S_{\mathrm{POVM}}^\star.
}
\]

这说明 DECA 对 commuting class structure 不是近似，而是精确最优。

### 9.1 多类 Spectral-DECA

对多类定义共同基下的对角近似

\[
\widehat A_c(P)
=
P\operatorname{diag}\!\left(P^\dagger A_cP\right)P^\dagger
\]

和谱亲和度

\[
q_c^{\mathrm{spec}}(x)
=
\operatorname{Tr}(\widehat A_c(P)\rho_x).
\]

同一次 computational-basis 采样可通过不同的 class weights 重用来估计
所有 \(q_c^{\mathrm{spec}}\)。对任意纯态，

\[
\left|
\operatorname{Tr}\!\left((A_c-\widehat A_c)\rho_x\right)
\right|
\le
\|O_c\|_2
\le
\|O_c\|_F.
\]

所以 joint-diagonalization residual 不只控制 single-shot POVM gap，也控制
谱亲和度对完整 class-operator score 的逐样本近似误差。

### 9.2 维度—类别数限制

一个 \(d\) 维 PVM 最多有 \(d\) 个非零正交 outcomes。将 outcomes hard-map
到 \(K\) 个类别后，非零 class effects 的数量仍不超过 \(d\)。因此

\[
\boxed{
K>d
\quad\Longrightarrow\quad
\text{至少 }K-d\text{ 个类别没有独立的非零 PVM effect}.
}
\]

这不是优化失败，而是结构容量限制。可选修复为：

1. 提高编码维度，使 \(d\ge K\)；
2. 使用有 \(K\) 个 effects 的一般 POVM/PGM；
3. 通过 Naimark dilation 增加 outcome ancilla；
4. 使用层次化多次测量，而不是单次 flat \(K\)-class PVM。

## 10. 定理四：joint-diagonalization residual 控制 oracle gap

对任意基 \(P\)，定义

\[
\widetilde A_c=P^\dagger A_cP
=D_c+O_c,
\]

其中

\[
D_c=\operatorname{diag}(\widetilde A_c),
\qquad
O_c=\operatorname{offdiag}(\widetilde A_c).
\]

令

\[
R_{\mathrm{off}}(P)
=
\left(
\sum_c\|O_c\|_F^2
\right)^{1/2}.
\]

取不受限最优 POVM 在 \(P\) 基中的表示 \(\{F_c\}\)。其对角部分仍形成
一个有效 stochastic decoder，而固定基的 hard MAP decoder 不差于任意
soft diagonal decoder。因此

\[
\begin{aligned}
S_{\mathrm{POVM}}^\star-S_{\mathrm{DECA}}(P)
&\le
\sum_c|\operatorname{Tr}(F_cO_c)|\\
&\le
\left(\sum_c\|F_c\|_F^2\right)^{1/2}
\left(\sum_c\|O_c\|_F^2\right)^{1/2}.
\end{aligned}
\]

由于 \(0\preceq F_c\preceq I\)，

\[
\|F_c\|_F^2
=\operatorname{Tr}(F_c^2)
\le\operatorname{Tr}(F_c),
\]

且

\[
\sum_c\operatorname{Tr}(F_c)=d,
\]

所以

\[
\boxed{
0\le
S_{\mathrm{POVM}}^\star-S_{\mathrm{DECA}}(P)
\le
\sqrt d\,R_{\mathrm{off}}(P).
}
\]

### 含义

- 近似联合对角化残差是一个有理论意义的 model-mismatch 指标；
- residual 为零时恢复 commuting exactness；
- 可在训练前判断“共享本征基”是否合理；
- SDP oracle gap 与 residual 可共同作为实验指标。

这个上界可能较松，但它是可计算、可证伪的第一版保证。

## 11. 多分类算法：Jacobi-DECA

目标为

\[
\max_{P^\dagger P=I}
F(P)
=
\sum_j\max_c p_j^\dagger A_cp_j.
\]

这是非凸、分段光滑问题。使用两个都有解析解的 coordinate steps。

### 11.1 Assignment step

固定 \(P\)：

\[
z_j\leftarrow
\arg\max_c p_j^\dagger A_cp_j.
\]

该步全局最优地更新所有 component labels，不降低目标。

### 11.2 Pair-rotation step

固定 labels。选择两个 component \(p_j,p_k\)，其 labels 分别为
\(c=z_j\)、\(r=z_k\)。

若 \(c=r\)，在该二维子空间内旋转不改变目标。

若 \(c\ne r\)，令

\[
B=A_c-A_r,
\qquad
Q=[p_j,p_k],
\qquad
\widehat B=Q^\dagger BQ\in\mathbb C^{2\times2}.
\]

旋转后的第一方向 \(p'_j=Qv\) 对应类别 \(c\)，第二正交方向对应类别
\(r\)。该 pair 对目标的可变部分为

\[
{p'_j}^\dagger(A_c-A_r)p'_j
=v^\dagger\widehat Bv.
\]

因此最优 \(v\) 是 \(\widehat B\) 最大本征值的单位本征向量；另一列取其
正交补。每次 pair rotation 都有闭式 \(2\times2\) eigensolution，且不降低
目标。

### 11.3 单调性

- assignment step 不降低 \(F\)；
- pair update 不降低 \(F\)；
- \(F(P)\le1\)。

所以目标值序列单调有界并收敛。算法不需要 learning rate。迭代点不保证
全局最优，但多启动与 SDP oracle 可量化其质量。

### 11.4 初始化

建议至少比较：

1. identity；
2. random Haar orthogonal/unitary；
3. pooled-state eigenbasis；
4. pairwise-contrast initializer：

   \[
   M=\sum_{c<r}(A_c-A_r)^2
   \]

   的本征基；
5. approximate joint diagonalization initializer。

选择训练目标最高的启动结果。

## 12. 与原 ECA 的关系

原 ECA 类别得分

\[
q_c(x)
=
\sum_jL_{jc}|p_j^\dagger\phi(x)|^2
=
\operatorname{Tr}(E_c\rho_x),
\]

其中

\[
E_c=P\operatorname{diag}(L_{:c})P^\dagger.
\]

如果

\[
L_{jc}\ge0,\qquad \sum_cL_{jc}=1,
\]

则它严格是 commuting POVM。

DECA 修正了三个问题：

1. 用 row simplex 代替独立 sigmoid，保证概率归一；
2. 在 minimum-error objective 下解析消去 \(L\)，得到 hard allocation；
3. 用严格 orthogonal/unitary \(P\)，不在 skew generator 内加入破坏正交性的
   diagonal scaling。

## 13. 经典分类与单次量子判别的区别

训练目标

\[
\sum_c\pi_c\operatorname{Tr}(E_c\rho_c)
\]

严格等于：

> 从 class ensemble 随机取一个状态，只允许一次测量时的平均正确率。

对经典数据，常见推理会重复计算全部概率并取

\[
\arg\max_c\operatorname{Tr}(E_c\rho_x).
\]

这时：

- single-shot success optimality 不等于 deterministic test accuracy optimality；
- Helstrom/SDP 是 class-average state discrimination 的 oracle；
- 泛化性能仍必须由 held-out experiments 证明。

论文必须明确区分：

1. expected single-shot success；
2. exact-probability argmax accuracy；
3. finite-shot quantum accuracy。

还必须区分第四个量：

4. 保留本征值大小的 Spectral-DECA observable expectation。

本项目的受控实验确认了这种区分的必要性：纯 covariance-signal 数据上，
Spectral-DECA-amplitude 的重复五折准确率为 \(0.786\pm0.014\)，而
PVM-DECA-amplitude 为 \(0.623\pm0.038\)。这不是与 Helstrom 最优性矛盾，
因为二者优化的 operational loss 不同。

## 14. 编码

### 14.1 Amplitude encoding

\[
\phi(x)=\frac{x}{\|x\|_2}.
\]

优点：

- 与 ECA 原始平方投影完全一致；
- 量子电路定义直接。

限制：

- 丢失幅度；
- \(\phi(x)\) 与 \(\phi(-x)\) 表示同一 density；
- 只产生齐次二次边界。

### 14.2 Affine amplitude lift

\[
\phi_\tau(x)
=
\frac{[x^\top,\tau]^\top}
{\sqrt{\|x\|_2^2+\tau^2}}.
\]

对应 density 中包含

\[
\begin{bmatrix}
xx^\top & \tau x\\
\tau x^\top & \tau^2
\end{bmatrix},
\]

因此 measurement score 同时包含：

- quadratic terms；
- linear terms；
- bias-like term。

\(\tau\) 必须仅在训练 fold 内选择。

### 14.3 Stereographic encoding

已有 HQC 工作使用 inverse stereographic projection。它应作为 prior-art
baseline，而不是 DECA 新贡献。

## 15. 量子电路

### 15.1 DECA/PVM：无 ancilla

把维度 pad 到

\[
D=2^{\lceil\log_2d\rceil}.
\]

设 DECA basis 为 \(P\)。对输入状态：

1. 准备 \(|\phi(x)\rangle\)；
2. 施加 \(P^\dagger\)；
3. 在 computational basis 测量；
4. 将 outcome \(j\) 通过 \(z_j\) 映射到类别。

Born probability 为

\[
\Pr(j\mid x)=|p_j^\dagger\phi(x)|^2.
\]

二分类时，这个电路精确实现 Helstrom PVM。

### 15.1.1 Spectral-DECA：同一基、重复 shots

Spectral-DECA 使用相同的状态制备、\(P^\dagger\) 与 computational-basis
measurement。区别只在 classical post-processing：PVM-DECA 把 outcome
hard-map 到类别，Spectral-DECA 则按 \(\lambda_j\) 或 \(a_{jc}\) 加权并对
多次 shots 求平均。因此：

- 不增加 ancilla；
- 不增加 measurement bases；
- 不再是 single-shot classifier；
- shot 数由目标置信度与测试样本 margin 决定。

### 15.2 一般 SDP POVM：Naimark dilation

给定 effects \(\{E_c\}\)，令

\[
M_c=\sqrt{E_c},
\qquad
V=
\begin{bmatrix}
M_1\\
\vdots\\
M_K
\end{bmatrix}.
\]

因为

\[
V^\dagger V=\sum_cE_c=I,
\]

\(V\) 是 isometry，可补全为 enlarged Hilbert space 上的 unitary \(W\)。

电路：

1. system 准备 \(|\phi(x)\rangle\)；
2. outcome ancilla 初始化为 \(|0\rangle\)；
3. 施加 \(W\)；
4. 测量 ancilla，得到类别 \(c\)。

该电路需要额外

\[
\lceil\log_2K\rceil
\]

个 outcome qubits，并需要更大的任意 unitary。

### 15.3 物理贡献

DECA 的硬件价值不是声称量子加速，而是：

- 用 ancilla-free PVM 近似一般 POVM；
- 对二分类与 commuting multi-class 保证 exact；
- 对一般多分类用 \(R_{\mathrm{off}}\) 和 SDP gap 量化代价；
- 将测量复杂度、ancilla 数和 classification success 放在同一 trade-off 中。

## 16. 预期应用

### 16.1 原生量子数据：首要应用

- qubit readout state discrimination；
- quantum sensing pattern discrimination；
- quantum communication receiver design；
- device calibration 和 drift monitoring；
- 受限 measurement hardware 上的 POVM approximation。

这些任务避免 classical amplitude loading，是最可信的量子应用。

### 16.2 类条件二阶结构显著的经典信号

- EEG/BCI 和 covariance features；
- radar/sonar/RF modulation；
- vibration-based fault diagnosis；
- hyperspectral spectral signatures；
- zero-mean texture 与 scattering features；
- few-shot class prototypes in normalized embedding spaces。

### 16.3 不应优先声称

- 任意 tabular 数据上普遍优于 boosted trees；
- 仅凭少样本就具有隐私；
- 仅凭 \(\log d\) qubits 就获得端到端指数加速；
- 未做硬件实验就宣称功耗或 latency 优势。

## 17. 已完成的实验证据

### 17.1 理论 sanity checks

- binary DECA success 与 Helstrom formula 数值一致；
- binary DECA 与 SDP oracle 一致；
- commuting multi-class gap 为数值零；
- noncommuting family 中检验 gap、residual 与 commutator 的关系；
- theorem-4 bound 从不被违反；
- Jacobi objective 单调不下降。

### 17.2 量子模拟

- PVM 电路 shot frequencies 对齐解析 Born probabilities；
- Naimark circuit 对齐 SDP POVM probabilities；
- binary PVM 不使用 ancilla；
- trine-state multi-class 展示 general POVM 对 PVM 的真实优势；
- finite-shot accuracy/variance 曲线。

### 17.3 经典数据

- covariance-only synthetic；
- mean-only synthetic；
- commuting vs noncommuting covariance sweep；
- Iris、Wine、Breast Cancer、Digits 等可复现数据；
- 与 logistic、LDA、QDA、RBF/poly SVM、CSP/HQC、PGM、SDP-POVM 比较；
- repeated stratified CV、mean/CI、balanced accuracy、NLL；
- 训练/推理时间与参数/存储；
- amplitude vs affine vs stereographic encoding ablation。

截至 2026-07-27：

- 30 个随机 binary trials 的闭式解误差不超过
  \(8.9\times10^{-16}\)，与 SDP 的最大 gap 为
  \(2.0\times10^{-8}\)；
- 16 个 commuting multi-class trials 与 SDP 的最大 gap 为
  \(1.45\times10^{-8}\)；
- 72 个 noncommuting trials 没有违反
  \(\sqrt dR_{\mathrm{off}}\) bound，但 commutator 与 gap 并不单调相关；
- Qiskit Aer PVM/Naimark 采样的最大 total-variation error 为 \(0.0073\)；
- trine ensemble 的一般 POVM success 为 \(2/3\)，比最佳受限 PVM 高
  \(0.04466\)；
- 完成 10 数据集、11 方法、2 repeats \(\times\) 5 folds，共 1,100 次
  外层拟合。DECA 在受控二阶数据上验证机制，但在多数通用公开数据上不优于
  tuned-free SVM/forest baselines；Letter 的 \(K=26>d=17\) 实验直接暴露
  PVM 容量限制。

## 18. 候选论文贡献陈述

在实验支持后，可以写成：

1. We recast ECA as minimum-error discrimination under a commuting-measurement
   constraint, clarifying why a shared unitary is redundant for unrestricted
   POVMs but meaningful for measurement compression.
2. We prove that the optimal class decoder is hard, derive a closed-form binary
   solution equal to the Helstrom optimum, and establish exactness for
   commuting multi-class operators.
3. We bound the loss relative to the optimal POVM by the joint-diagonalization
   residual and propose a monotone Jacobi algorithm whose pair updates have
   analytical \(2\times2\) solutions.
4. We compile DECA into an ancilla-free quantum circuit, compare it with a
   Naimark-dilated SDP oracle, and evaluate statistical and circuit-resource
   trade-offs on quantum-state and classical classification tasks.

其中第 2 项的 Helstrom 内容必须引用已有理论；原创点是 ECA constrained
measurement formulation、hard-decoder theorem、multi-class结构、gap bound 和
Jacobi-DECA 组合。
