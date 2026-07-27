# ECA 深度研究分析：从“最大类间差异”到可验证的判别能量分解

> 研究对象：Eigen-Component Analysis（ECA）、uECA，以及相关的 Ising Clustering
>
> 整理日期：2026-07-27
>
> 性质：研究分析与下一版算法设计文档，不是对现有论文结论的无条件背书

## 0. 证据范围与来源说明

本分析综合了以下材料：

1. 原始预印本 [`2003.10199v3.pdf`](./2003.10199v3.pdf)：
   *Eigen Component Analysis: A Quantum Theory Incorporated Machine Learning Technique to Find Linearly Maximum Separable Components*。
2. ISCAS 2025 论文工程：
   [`_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model`](./_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/)。
3. TCAS-II 候选投稿工程（保留在本机；因与前述工程的主要文件重复，且不能验证为系统最终上传版本，未纳入公开仓库）。
4. TCAS-II-24755-2025 的决定信与全部评审意见（保留在本机；公开仓库只保留匿名化研究总结，不发布编辑通信和收件人信息）。
5. ICIP 方向草稿：
   [`_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/main.tex`](./_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/main.tex)。
6. 先前与 ChatGPT 的探索性讨论：
   [`1. 更 general 的数学：ECA 代表什么通用概念？.md`](./1.%20更%20general%20的数学：ECA%20代表什么通用概念？.md)。
7. 公开的 [`lachlanchen/eca`](https://github.com/lachlanchen/eca) 实现，以及本文末列出的机器学习、统计学、优化和量子信息原始文献。

需要特别注明：

- 本地 ISCAS 与 TCAS 两个工程中的主要 ECA `.tex`、`.bib` 文件目前是逐字节相同的，且本地 `conference_101719.pdf` 是 IEEE 空白模板。因此，无法仅凭本地目录精确恢复当时 TCAS-II 系统中最终上传的 PDF。本文把这些文件称为“候选投稿稿件”，并以决定信和评审意见为确切投稿证据。
- ISCAS 论文的公开发表版本可由 [HKU 作者页面](https://www.eee.hku.hk/~elam/research/pub-dl.html)及其[公开 PDF](https://www.eee.hku.hk/optima/pub/conference/2505_ISCASa.pdf)确认，DOI 为 `10.1109/ISCAS56072.2025.11044249`。
- 既有 ChatGPT 对话是重要的研究笔记，但不是独立学术证据；其中的观点需要由推导、代码或文献重新验证。本文已经对关键主张作了这种复核。

---

## 1. 先给结论

### 1.1 你的原始直觉是有价值的，但需要换一个精确表述

你的核心问题可以表述为：

> PCA 寻找总体方差最大的方向；但用于分类时，更重要的不是一个方向“变化有多大”，而是不同类别在这个方向上的统计响应“相差有多大”。能否直接学习这种判别方向？

这是一个正确而重要的问题。但“最大类间差异”本身不是全新的研究目标；Fisher LDA、Maximum Margin Criterion（MMC）、Common Spatial Patterns（CSP）、Neighborhood Component Analysis（NCA）、metric learning 等都以不同方式追求类别可分性。

ECA 最有希望的独特位置不是泛泛地宣称“最大化 separability”，而是：

> 学习一个共享正交基，使每个类别在该基上的方向性能量（squared projection energy）形成稀疏、稳定、差异显著的谱；分类器则由共享本征基上的类别能量算符组成。

这把 ECA 放到了几个成熟方向的交叉点：

- 判别降维；
- 类条件二阶矩与协方差比较；
- 近似联合对角化；
- 类子空间分类；
- 共享本征向量的结构化二次分类器；
- 可交换量子测量的经典实现。

这个定位比“quantum-inspired linear model”更准确，也更容易形成可验证的理论贡献。

### 1.2 ECA 不是输入空间中的线性分类器

若

\[
\psi=P^\top x,\qquad
z_j=\psi_j^2=(p_j^\top x)^2,
\]

并令类别 \(c\) 的得分为

\[
s_c(x)=\sum_j L_{jc}z_j,
\]

则

\[
s_c(x)
=\sum_j L_{jc}(p_j^\top x)^2
=x^\top P\operatorname{diag}(L_{:c})P^\top x
=x^\top Q_cx.
\]

所以 ECA 是一个结构化的齐次二次分类器，其中

\[
Q_c=P\operatorname{diag}(L_{:c})P^\top.
\]

所有 \(Q_c\) 共享同一个本征基 \(P\)，因此彼此可交换：

\[
Q_cQ_d=Q_dQ_c.
\]

类 \(c\) 与类 \(d\) 的决策边界是

\[
x^\top(Q_c-Q_d)x=0,
\]

它一般是二次锥面，而不是超平面。

因此最准确的说法是：

> ECA 在能量特征
> \(\phi_P(x)=((p_1^\top x)^2,\ldots,(p_r^\top x)^2)\)
> 上是线性的，但在原输入 \(x\) 上是二次的。

这不是措辞上的小修正，而是论文理论定位的中心。

### 1.3 一个正交变换本身不可能把线性不可分数据变成线性可分

假设正交变换后存在超平面

\[
w^\top P^\top x+b=0
\]

能够分开数据，那么令 \(v=Pw\)，原空间中就有

\[
v^\top x+b=0.
\]

因此，可逆线性变换保持线性可分性。ECA 的额外表达能力来自逐分量平方，而不是来自旋转 \(P\) 本身。

相应地，“find linearly maximum separable components”只能理解为“在非线性能量特征空间里获得更可分的表示”，不能理解为正交旋转创造了原本不存在的线性可分性。

### 1.4 最值得发展的新算法

建议把下一版算法定义为 **Discriminative Energy Component Analysis（DECA）**，或保留 ECA 名字并使用副标题：

> ECA: Class-Contrast Joint Diagonalization of Conditional Energy Operators

核心目标不是最大化总方差，而是最大化类别条件能量谱之间的差异：

\[
\max_{P^\top P=I}
\sum_c\pi_c
\left\|
\operatorname{diag}\!\left(P^\top(R_c-\bar R)P\right)
\right\|_2^2,
\]

其中

\[
R_c=\mathbb E[xx^\top\mid y=c],
\qquad
\bar R=\sum_c\pi_cR_c.
\]

这个目标有清晰的统计意义、矩阵意义和物理意义：

- 统计上：寻找不同类别二阶响应差异最大的方向；
- 矩阵上：近似联合对角化中心化后的类二阶矩；
- 物理上：寻找使不同类“态”具有最可区分测量能量谱的可交换测量基。

---

## 2. ECA 的正确数学对象

### 2.1 从样本到方向性能量

设 \(x\in\mathbb R^m\)，通常先做 \(\|x\|_2=1\)。令

\[
P=[p_1,\ldots,p_m]\in O(m),
\qquad P^\top P=I.
\]

变换后的坐标为

\[
\psi=P^\top x.
\]

ECA 不直接使用有符号坐标 \(\psi_j\)，而使用

\[
z_j=|\psi_j|^2=(p_j^\top x)^2.
\]

若 \(P\) 正交且 \(x\) 单位归一化，

\[
\sum_j z_j=\|P^\top x\|_2^2=\|x\|_2^2=1.
\]

因此 \(z\) 位于概率单纯形上，可解释为样本能量在正交方向之间的分配。

这个映射具有三个重要性质：

1. **丢弃符号**

   \[
   z_P(x)=z_P(-x).
   \]

   因此纯 ECA 无法区分互为相反数的样本。

2. **引入二阶非线性**

   它等价于对 rank-one 二阶张量 \(xx^\top\) 做线性测量：

   \[
   z_j
   =p_j^\top xx^\top p_j
   =\operatorname{Tr}(p_jp_j^\top xx^\top).
   \]

3. **只保留特定二次项**

   对固定 \(P\)，ECA 使用在 \(P\) 基下的平方坐标。它不是一个包含全部任意交叉项的自由二次模型；类别矩阵受到共享本征基的强结构约束。

### 2.2 共享本征基的二次分类器

令 \(L\in\mathbb R^{m\times K}\)，类别得分为

\[
s_c(x)=z^\top L_{:c}+b_c.
\]

则

\[
s_c(x)=x^\top Q_cx+b_c,
\qquad
Q_c=P\operatorname{diag}(L_{:c})P^\top.
\]

这给出一个清楚的模型层级：

| 模型 | 类别矩阵 \(Q_c\) | 表达能力与代价 |
|---|---|---|
| 线性 softmax | 无二次矩阵 | \(O(mK)\) 参数 |
| ECA | 对称且共享本征基 | \(O(m^2+mK)\)，结构化二次 |
| 低秩 ECA | \(P\in\mathrm{St}(m,r)\) | \(O(mr+rK)\) |
| 一般 QDA / quadratic softmax | 每类独立对称矩阵 | \(O(Km^2)\) |

ECA 的统计归纳偏置是：

> 不同类别可以赋予各方向不同权重，但这些方向对所有类别共享。

若数据中的类别二阶结构确实近似共享本征方向，这个约束可显著降低一般二次模型的方差；若不同类别的协方差强烈不对易，约束则会造成欠拟合。

### 2.3 硬分配时，ECA 是类子空间能量分类器

如果 \(L\) 的每一行是 one-hot，即每个方向只分配给一个类别，令

\[
\mathcal I_c=\{j:L_{jc}=1\}.
\]

则

\[
Q_c=\sum_{j\in\mathcal I_c}p_jp_j^\top
\]

是投影矩阵，类别得分为

\[
s_c(x)
=\sum_{j\in\mathcal I_c}(p_j^\top x)^2
=\|\Pi_cx\|_2^2.
\]

因此硬 ECA 选择“哪个类别子空间吸收了最多投影能量”。这与 nearest-subspace / projection-subspace 分类有直接关系。

若这些子空间构成正交直和，且 \(\|x\|=1\)，则最大化投影能量等价于最小化到类子空间的重构残差：

\[
\|x-\Pi_cx\|_2^2
=1-\|\Pi_cx\|_2^2.
\]

这是一种比“quantum neural network”更具体、也更容易与经典文献比较的解释。

### 2.4 softmax 分类下，非负 \(L\) 不是主要本质，`[0,1]` 边界才是

如果使用

\[
p(y=c\mid x)=\operatorname{softmax}_c(s(x)),
\]

那么对每个方向 \(j\)，同时给所有类别权重加上相同常数 \(a_j\)：

\[
L_{jc}\leftarrow L_{jc}+a_j
\]

只会给所有类别 logit 加上相同的输入相关项

\[
\sum_j a_jz_j,
\]

softmax 概率不变。因此，在共享本征基模型中，可以逐行平移 \(L\) 使其非负，而不改变分类概率。

所以“\(L\ge 0\)”本身并不构成很强的限制；真正的限制来自：

- 把 \(L\) 压到固定区间 \([0,1]\)；
- 直接把未归一化得分称为概率；
- 使用硬阈值而不给出离散化与校准的理论。

---

## 3. “最大类间差异”应怎样严格定义

### 3.1 PCA 为什么不等于判别特征

PCA 最大化

\[
\operatorname{Var}(p^\top x)=p^\top Rp,
\]

其中 \(R\) 是总体协方差或二阶矩。它不使用标签。

一个方向可能有很大方差，但这种方差完全来自每个类别内部的噪声；也可能有一个方向总体方差不大，但不同类别在该方向上的统计量非常稳定且差异明显。PCA 不区分这两种情况。

不过，“类间差异”至少有三种不同含义：

1. **均值差异**

   \[
   p^\top(\mu_c-\mu_d).
   \]

   这是 LDA、MMC 等方向最熟悉的对象。

2. **方差或能量差异**

   \[
   p^\top(R_c-R_d)p.
   \]

   这是 ECA 平方响应、CSP、协方差判别所对应的对象。

3. **整个投影分布的差异**

   例如 MMD、Wasserstein 距离、KL divergence 或分类风险。

ECA 的平方投影天然对应第二种，即**类条件二阶能量差异**，不是一般意义下的全部分布差异。

### 3.2 类条件二阶矩

定义

\[
R_c=\mathbb E[xx^\top\mid y=c].
\]

注意

\[
R_c=\Sigma_c+\mu_c\mu_c^\top.
\]

因此二阶矩同时包含类内协方差与未中心化均值信息。如果对每个类分别中心化，剩下的才是纯协方差差异；如果只做全局中心化，均值差异仍会进入 \(R_c\)。

对方向 \(p_j\)，类别 \(c\) 的平均 ECA 能量为

\[
e_{cj}
=\mathbb E[(p_j^\top x)^2\mid y=c]
=p_j^\top R_cp_j.
\]

所以每个类有一个能量谱

\[
e_c(P)=\operatorname{diag}(P^\top R_cP).
\]

你的原始思想可以精确写成：

> 学习 \(P\)，使不同类的 \(e_c(P)\) 尽可能不同，同时使同类样本的能量响应足够稳定。

### 3.3 类能量对比目标

令

\[
\bar R=\sum_c\pi_cR_c,
\qquad
\Delta_c=R_c-\bar R.
\]

定义

\[
\boxed{
J_{\mathrm{contrast}}(P)
=
\sum_c\pi_c
\left\|
\operatorname{diag}(P^\top\Delta_cP)
\right\|_2^2
}
\]

并最大化

\[
\max_{P^\top P=I}J_{\mathrm{contrast}}(P).
\]

展开后，

\[
J_{\mathrm{contrast}}(P)
=\sum_{c,j}\pi_c
\left(p_j^\top\Delta_cp_j\right)^2.
\]

它直接奖励：

- 某一方向对某些类别具有高于平均的能量；
- 对另一些类别具有低于平均的能量；
- 类别能量轮廓在方向上形成可解释的差异。

这里使用平方非常重要。若简单地在完整正交基上最大化

\[
\sum_j p_j^\top Sp_j=\operatorname{Tr}(S),
\]

目标对所有完整正交基都是常数，根本学不到方向。很多“最大总类差异”的朴素写法会掉入这个 trace invariance 陷阱。平方对比、非线性凸函数，或只选 \(r<m\) 个方向，才能使基的选择非平凡。

### 3.4 它等价于近似联合对角化

由于 Frobenius 范数在正交变换下不变，

\[
\|\Delta_c\|_F^2
=
\|P^\top\Delta_cP\|_F^2.
\]

而任意矩阵可分解为对角与非对角部分：

\[
\|P^\top\Delta_cP\|_F^2
=
\|\operatorname{diag}(P^\top\Delta_cP)\|_2^2
+
\|\operatorname{offdiag}(P^\top\Delta_cP)\|_F^2.
\]

所以最大化 \(J_{\mathrm{contrast}}\) 等价于最小化

\[
\sum_c\pi_c
\|\operatorname{offdiag}(P^\top\Delta_cP)\|_F^2.
\]

换言之：

> DECA 正在寻找一个共同基，使所有中心化类二阶矩尽可能同时对角化。

如果所有 \(\Delta_c\) 两两可交换，

\[
[\Delta_c,\Delta_d]
=\Delta_c\Delta_d-\Delta_d\Delta_c=0,
\]

那么这些实对称矩阵可以被同一个正交矩阵精确同时对角化，并达到目标上界

\[
J_{\max}
=\sum_c\pi_c\|\Delta_c\|_F^2.
\]

如果它们不对易，则无法精确共享本征基；最优目标与残余非对角能量直接量化“共享本征基假设”的失配程度。

这是 ECA 可以建立的一个很好的理论核心：

- “commuting” 不再只是量子术语；
- 它对应一个可检验的统计假设；
- joint-diagonalization residual 可以成为模型诊断量；
- 非对易程度可以决定是否需要一般化到多个基或一般 POVM。

### 3.5 更贴近分类风险的类对 margin

只比较类均值能量还不够，因为同类内部也可能高度不稳定。定义能量特征

\[
z_P(x)=(P^\top x)\odot(P^\top x),
\]

类条件均值和协方差

\[
\mu_c^z=\mathbb E[z_P(x)\mid c],
\qquad
\Sigma_c^z=\operatorname{Cov}(z_P(x)\mid c).
\]

可定义类对能量 margin：

\[
M_{cd}(P)
=
\frac{
\|\mu_c^z-\mu_d^z\|_2^2
}{
\operatorname{Tr}(\Sigma_c^z+\Sigma_d^z)+\varepsilon
}.
\]

两种聚合方式代表不同研究目标：

#### 平均类对分离

\[
\max_P\sum_{c<d}w_{cd}M_{cd}(P).
\]

它优化总体平均表现，但可能牺牲难分类的类别对。

#### 最坏类对分离

\[
\max_P\min_{c<d}M_{cd}(P).
\]

这最接近你所说的“max difference across classes”中更有力量的一版：不是让容易的类别差得更远，而是提高最难类别对的最低 margin。

为便于梯度优化，可用 soft minimum：

\[
\operatorname{softmin}_\tau(M)
=-\tau\log\sum_{c<d}\exp(-M_{cd}/\tau).
\]

当 \(\tau\to0\) 时，它逼近最小类对 margin。

### 3.6 推荐的监督目标

一个可实际训练的目标是

\[
\begin{aligned}
\mathcal L
=&\quad
\mathcal L_{\mathrm{CE}}
\bigl(
\operatorname{softmax}(W^\top z_P(x)+b),y
\bigr) \\
&-\alpha J_{\mathrm{contrast}}(P)
-\gamma\operatorname{softmin}_{c<d}M_{cd}(P)\\
&+\beta\Omega(W)
+\eta\Omega_{\mathrm{sparse}}(W).
\end{aligned}
\]

其中 \(P\in\mathrm{St}(m,r)\)，即

\[
P^\top P=I_r.
\]

这使模型可以：

- 用交叉熵直接优化预测；
- 用 \(J_{\mathrm{contrast}}\) 保证学到的方向确实有类能量意义；
- 用最坏类对 margin 避免只优化容易类别；
- 用 \(r<m\) 实现真正的判别降维；
- 用稀疏正则得到可解释的 component-class allocation。

---

## 4. 与已有方法的关系：什么已知，什么可能是 ECA 的贡献

| 方法 | 主要统计量 | 是否监督 | 典型目标 | 与 ECA 的关系 |
|---|---:|---:|---|---|
| PCA | 总体协方差 | 否 | 最大总方差 | ECA 不应以 PCA 方差为目标 |
| Fisher LDA | 类均值、类内散度 | 是 | 最大类间/类内散度比 | 已经系统研究“判别方向” |
| MMC | \(S_b-S_w\) | 是 | 最大 margin criterion | 最接近“差而不是方差”的经典措辞 |
| CSP | 两类协方差 | 是 | 一类方差大、另一类小 | 与 ECA 的平方投影能量尤其接近 |
| Common Principal Components | 多组协方差 | 通常为统计估计 | 共享本征向量 | ECA 共享类本征基的直接前身 |
| Joint diagonalization | 多个矩阵 | 可监督/无监督 | 同时减少非对角项 | DECA 的自然数学工具 |
| QDA | 每类均值与协方差 | 是 | 类别高斯似然 | 一般二次边界，参数更多 |
| Nearest subspace classifier | 类子空间 | 是 | 最小重构残差 | 硬 ECA 等价于比较投影能量 |
| NCA / LMNN | 局部距离与邻域 | 是 | 提高近邻分类率/margin | 处理局部结构，ECA 当前是全局基 |
| Polynomial SVM | 二阶或更高核特征 | 是 | 最大 margin | 是必须比较的二次模型基线 |
| ECA / DECA | 类条件方向性能量 | 是 | 共享基的判别能量谱 | 候选独特贡献 |

### 4.1 不能作为新颖性核心的表述

以下主张过于宽泛，容易被现有文献覆盖：

- “PCA maximizes variance; we maximize separability”；
- “找到 discriminative features”；
- “orthogonal transform for classification”；
- “用二阶信息做分类”；
- “学习 class-specific subspace”；
- “quantum probability inspires classical ML”。

### 4.2 更有防御力的新颖性表述

较强的贡献可以由下面四部分组成：

1. **模型类**

   提出一个共享本征基的多类二次能量分类器，即一族可交换类别算符。

2. **判别目标**

   直接最大化类条件二阶能量谱的对比度或最坏类对 margin，而不是总体方差。

3. **理论连接**

   证明目标等价于类二阶矩的监督近似联合对角化，并以 commutator / residual 衡量模型适用性。

4. **可扩展算法**

   在低秩 Stiefel 流形上优化，给出明确参数量、训练复杂度、推理复杂度和统计泛化实验。

如果四者都完成，ECA 就不再只是“受量子启发的一个网络层”，而是一个清楚、可比较、可推广的机器学习模型族。

---

## 5. 物理与量子信息解释：保留什么，删除什么

### 5.1 严格成立的密度矩阵表达

对单位归一化的实向量 \(x\)，定义纯态密度矩阵

\[
\rho_x=xx^\top,
\qquad
\rho_x\succeq0,
\quad
\operatorname{Tr}(\rho_x)=1.
\]

方向投影算符

\[
\Pi_j=p_jp_j^\top.
\]

则 ECA 能量为

\[
z_j
=\operatorname{Tr}(\Pi_j\rho_x)
=|\langle p_j,x\rangle|^2.
\]

若 \(P\) 正交，

\[
\Pi_j\Pi_\ell=0\quad(j\ne\ell),
\qquad
\sum_j\Pi_j=I.
\]

这就是一个投影值测量（PVM）的 Born-rule 概率。

### 5.2 soft \(L\) 对应可交换 POVM 的条件

定义类别 effect：

\[
E_c
=\sum_jL_{jc}\Pi_j
=P\operatorname{diag}(L_{:c})P^\top.
\]

若

\[
L_{jc}\ge0,
\qquad
\sum_cL_{jc}=1\quad\forall j,
\]

则

\[
E_c\succeq0,
\qquad
\sum_cE_c=I.
\]

类别概率为

\[
q_c(x)=\operatorname{Tr}(E_c\rho_x).
\]

因此：

- \(L\) 每行 one-hot：PVM 的粗粒化，方向被硬分到类别；
- \(L\) 每行是概率分布：可交换 POVM；
- 所有 \(E_c\) 共享 \(P\)：测量 effects 两两可交换。

这是当前 ECA 最干净、最严格的量子对应。

### 5.3 当前 sigmoid \(L\) 不保证是类别概率

现有稿件和代码使用逐元素 sigmoid：

\[
L_{jc}=\sigma(\ell_{jc})
\]

或其硬阈值版本。但 sigmoid 只保证

\[
0<L_{jc}<1,
\]

不保证

\[
\sum_cL_{jc}=1.
\]

因此即使 \(\sum_jz_j=1\)，也通常有

\[
\sum_cq_c
=\sum_jz_j\sum_cL_{jc}
\ne1.
\]

这时 \(q_c\) 不能直接称为 categorical probability。

公开实现进一步把 `class_scores = (psi**2) @ L` 送入 PyTorch `CrossEntropyLoss`。该损失会再次应用 log-softmax，所以代码实际上把这些量当 **logits**，而不是直接使用 Born-rule probability。

建议在下面两条路线中明确选择一条。

#### 路线 A：机器学习优先

\[
s=z^\top L+b,\qquad
p(y\mid x)=\operatorname{softmax}(s).
\]

- 把 \(s\) 称为 energy logits；
- \(L\) 可连续、可带符号或仅作可解释约束；
- 不宣称 \(s\) 本身是 Born probability。

#### 路线 B：物理一致性优先

\[
L_{j:}=\operatorname{softmax}(\ell_{j:}),
\qquad
q_c=\sum_jz_jL_{jc}.
\]

- \(\sum_cq_c=1\)；
- \(E_c\) 构成 POVM；
- 训练使用 \(-\log q_y\)，而不是对 \(q\) 再做 softmax；
- 可加入温度参数控制软硬分配。

不能一边使用路线 A 的代码，一边使用路线 B 的概率推导。

### 5.4 二类情形与 Helstrom 测量

把每类样本的平均密度矩阵定义为

\[
\rho_c=\mathbb E[xx^\top\mid y=c].
\]

对二分类，先验为 \(\pi_1,\pi_2\)，令

\[
\Delta=\pi_1\rho_1-\pi_2\rho_2.
\]

若目标是最大化两类平均正确识别概率，并允许任意二元投影测量，则最优测量由 \(\Delta\) 的正、负本征子空间给出：

\[
E_1=\Pi_{\Delta>0},
\qquad
E_2=I-E_1.
\]

这就是量子态二元最小错误判别中的 Helstrom 结构。

它给 ECA 一个真正有数学含量的分析基线：

- 对二类平均态，最优投影可由一次特征分解得到；
- 神经训练的 ECA 可以与这个 closed-form surrogate 比较；
- 多类与样本级 discriminative loss 才需要更一般的优化。

但需要谨慎：

- 这是类平均密度矩阵的最优判别，不是现有 ECA 交叉熵目标的闭式解；
- 它只使用二阶矩；
- 若两类 \(\rho_c\) 相同但高阶分布不同，它无法区分。

### 5.5 从可交换到一般测量的模型层级

一个自然的理论路线是：

1. **硬 ECA / PVM**

   正交方向被离散分配给类别。

2. **soft ECA / commuting POVM**

   类别 effects 共享本征基。

3. **general PSD classifier / noncommuting POVM**

   各 \(E_c\succeq0\)，满足 \(\sum_cE_c=I\)，但不要求共享本征向量。

4. **一般 quadratic softmax**

   \(Q_c\) 可为任意对称矩阵，不要求 PSD 或 normalization。

这个层级允许用实验回答：

> ECA 的可交换约束究竟是有效归纳偏置，还是造成了显著性能损失？

### 5.6 不应宣称量子优势

当前算法在经典计算机上用 dense matrix exponential、矩阵乘法和梯度下降训练。仅仅把归一化向量叫作“state”、把平方投影叫作“measurement”，不能推出量子加速。

若未来声称量子优势，必须同时计算：

- 经典数据的 state preparation；
- 电路深度和门数；
- measurement shots；
- 类别概率估计误差；
- 训练和梯度估计成本；
- 读取输出的成本；
- 与最佳经典低秩算法的端到端比较。

现阶段最稳妥的说法是：

> ECA has a quantum-measurement-consistent interpretation under explicit normalization and POVM constraints.

而不是：

> ECA achieves quantum computational advantage.

---

## 6. 对现有论文和代码的关键技术审计

### 6.1 `diag(D)` 破坏正交性与能量守恒

论文写作倾向于令

\[
P=e^A,\qquad A^\top=-A,
\]

这时 \(P\in SO(m)\)，确实正交。

但公开代码实际使用

\[
A=A_{\mathrm{raw}}-A_{\mathrm{raw}}^\top+\operatorname{diag}(D),
\qquad
P=e^A.
\]

只要 \(D\ne0\)，就有

\[
A^\top\ne-A,
\]

从而一般不再有

\[
P^\top P=I.
\]

因为 skew 部分与 diagonal 部分通常不对易，不能把 \(e^A\) 简单解释为“正交旋转乘独立尺度”。

直接后果是：

- \(\sum_j|(P^\top x)_j|^2\) 不再等于 \(\|x\|^2\)；
- `psi**2` 不再自然归一；
- Born-rule / PVM 推导不成立；
- “orthogonal features” 的理论与实现不一致。

修复方法只能二选一：

1. **严格正交 ECA**

   设 \(D=0\)，在 Stiefel/orthogonal manifold 上优化。

2. **判别度量 ECA**

   明确允许

   \[
   P=UR
   \]

   或单独的正尺度 \(S\)，把模型解释为 metric learning / generalized energy model，并取消能量守恒的物理主张。

### 6.2 当前参数量报告与真实实现不一致

代码中可训练张量为：

- `A_raw`: \(m^2\)；
- `D`: \(m\)；
- `L_raw`: \(mK\)。

所以真实 raw trainable scalar 数量是

\[
m^2+m+mK.
\]

代码报告的却是

\[
\frac{m(m+1)}2+m+mK.
\]

对 Fashion-MNIST / MNIST 的 \(m=784,K=10\)：

- 代码真实 raw 参数：

  \[
  784^2+784+7840=623{,}280.
  \]

- 表格/代码报告：

  \[
  \frac{784\cdot785}{2}+784+7840=316{,}344.
  \]

- 严格正交 \(P\) 的 intrinsic degrees of freedom 加 \(L\)：

  \[
  \frac{784\cdot783}{2}+7840=314{,}776.
  \]

如果保留额外 \(m\) 个尺度，自由度则是 \(315{,}560\)，仍不等于报告值。

因此必须分别报告：

1. 优化器实际存储的 raw parameters；
2. 模型流形的 intrinsic degrees of freedom；
3. 推理时存储的 materialized \(P,L\)；
4. 训练时 optimizer state 和激活内存。

Reviewer 1 对参数优势的质疑是成立的。

### 6.3 dense matrix exponential 的训练成本

对 \(m\times m\) 矩阵，每次前向重新计算 dense matrix exponential，主成本通常是

\[
O(m^3).
\]

随后样本变换为

\[
O(Bm^2)
\]

每 batch，类别映射为

\[
O(BmK).
\]

因此即使参数数量少于完全自由的 \(K\) 个二次矩阵，训练并不自动快于 SVM 或 logistic regression。速度主张必须用：

- 统一硬件；
- 相同数据 split；
- 相同调参预算；
- wall-clock training；
- batch latency / throughput；
- 峰值显存；
- 推理是否预计算 \(P\)

来验证。

### 6.4 uECA 当前形式退化为普通线性 softmax

公开 uECA forward 可写为

\[
\operatorname{softmax}(XP L).
\]

令

\[
W=PL,
\]

则模型就是

\[
\operatorname{softmax}(XW).
\]

若没有额外约束使 \(P\) 和 \(L\) 分别具有不可替代的意义，它们不可辨识，且没有 ECA 的平方能量结构。

所以当前 uECA 并不是监督 ECA 的自然无监督版本，而更像一个过参数化的线性 softmax clustering map。

要建立真正的 uECA，应至少保留

\[
z=(P^\top x)^2
\]

并定义避免塌缩的聚类目标，例如：

\[
\mathcal L_{\mathrm{uECA}}
=
\underbrace{\sum_{ij}s_{ij}\|q_i-q_j\|^2}_{\text{局部平滑}}
-\lambda\underbrace{H(\bar q)}_{\text{全局均衡}}
+\gamma\underbrace{\frac1N\sum_iH(q_i)}_{\text{个体低熵}}
-\alpha J_{\mathrm{contrast-like}}.
\]

这里 \(q_i\) 是簇分配，\(\bar q=N^{-1}\sum_iq_i\)。局部平滑防止相近样本分开，全局熵防止全部落入同一簇，个体低熵促进确定分配。

### 6.5 “稀疏性来自正交性”尚未被证明

正交性只能保证

\[
\sum_jz_j=\|x\|^2.
\]

它不保证大多数 \(z_j\) 接近零。各向同性数据在任意正交基中都可能具有分散能量；低维流形也未必与一个全局线性基对齐。

同样，交叉熵和 sigmoid \(L\) 本身也不保证 component-class allocation 稀疏。

如果稀疏性是主要贡献，需要：

- 明确的 \(\ell_1\)、group lasso、entropy 或 hard-concrete 正则；
- Hoyer sparsity、Gini、participation ratio 等定量指标；
- 与随机正交基、PCA、LDA、MMC 的相同指标比较；
- 跨 seed 的 component stability；
- 在验证集上证明稀疏性与准确率、鲁棒性或硬件效率相关。

### 6.6 “无需标准化”不成立

正交变换保持欧氏距离和范数，但不使模型对不同输入 feature scales 自动不变。

如果某特征的单位从米变成毫米，其数值尺度改变，二阶能量会被放大 \(10^6\)；正交矩阵不会自动消除这个问题。

逐样本的 \(\ell_2\) normalization：

- 是一种 preprocessing；
- 消除了样本总体幅度；
- 不等于逐特征 standardization；
- 可能丢失有用的幅度信息。

下一版实验应分别比较：

- raw；
- per-feature z-score；
- robust scaling；
- per-sample \(\ell_2\) normalization；
- z-score 后再 \(\ell_2\) normalization。

### 6.7 “少量训练样本”等于隐私保护不成立

训练样本少只说明 sample efficiency 的候选现象，不构成 privacy guarantee。

隐私主张至少需要：

- threat model；
- central / local / federated 设置；
- 是否共享样本、梯度、二阶统计量；
- membership inference / reconstruction 攻击；
- differential privacy 参数 \((\varepsilon,\delta)\)；
- secure aggregation 或可信执行假设。

事实上 \(R_c=\sum_{i:y_i=c}x_ix_i^\top\) 这样的二阶充分统计量也可能泄露数据结构，不能因为“不传 raw data”就自动视为安全。

### 6.8 当前实验不能支撑广泛 superiority

ISCAS 结果展示了 ECA 在若干小型或下采样任务上的潜力，但证据边界应被准确描述：

- synthetic 和 Iris 任务过小，多个传统方法已饱和；
- Digits 上 ECA 与 kernel SVM 都达到 99.17%，但 ECA 报告参数 2784，kernel SVM 为 663；
- Fashion-MNIST 与 MNIST 的 ECA 报告参数为 316,344，远高于表中 kernel SVM 的数百个；
- 数据被大比例下采样，测试集也被下采样，结果方差可能很大；
- 未报告 repeated seeds、置信区间、nested CV；
- 未充分说明硬件、调参预算和计时方法；
- “fewer parameters and higher accuracy” 不能作为跨任务总括结论。

Reviewer 1 对参数与性能比较的质疑、Reviewer 2 对数据集复杂度与 scalability 的质疑，都需要通过新实验而不是文字辩护解决。

---

## 7. 推荐的可扩展参数化与优化

### 7.1 不必学习完整 \(m\times m\) 基

如果目标是找判别特征，最自然的是

\[
P\in\mathbb R^{m\times r},
\qquad
P^\top P=I_r,
\qquad
r\ll m.
\]

这就是 Stiefel manifold \(\mathrm{St}(m,r)\)。

模型为

\[
z=(P^\top x)^2\in\mathbb R^r,
\qquad
s=W^\top z+b.
\]

推理成本降为

\[
O(mr+rK),
\]

存储和参数规模也降为 \(O(mr+rK)\)。

Stiefel 矩阵的内在自由度是

\[
mr-\frac{r(r+1)}2.
\]

因此加上 \(W\in\mathbb R^{r\times K}\) 与偏置后，模型的 intrinsic dimension 为

\[
mr-\frac{r(r+1)}2+rK+K.
\]

这是比完整 \(m^2\) 变换更符合“feature extraction”的形式。

### 7.2 优化选项

可考虑以下方法：

1. **Riemannian gradient / retraction**

   在 Stiefel 流形上直接优化，理论最清楚。

2. **QR retraction**

   每步更新后做 thin QR；实现简单，但需处理符号与数值稳定。

3. **Cayley transform**

   用 skew-symmetric 更新保持正交约束。

4. **Givens rotations**

   适合稀疏可解释旋转和硬件实现。

5. **Householder products**

   用少量反射构建正交变换，参数/计算可控。

6. **joint diagonalization initialization**

   先对 \(\{\Delta_c\}\) 做 AJD，再用监督 loss 微调。

完整 matrix exponential 不是错误方法，但对于大 \(m\) 通常不是最经济的默认选择。

### 7.3 `SO(m)` 与 `O(m)` 的区别通常不重要

skew exponential 只能得到

\[
\det(P)=+1
\]

的特殊正交群 \(SO(m)\)，不能直接表示 determinant 为 \(-1\) 的反射。

但 ECA 使用平方投影，改变任意一列 \(p_j\) 的符号不改变 \(z_j\)。因此 \(O(m)\) 与 \(SO(m)\) 在这个模型中的表达差别通常可由列符号不变性消除，不是主要限制。

---

## 8. 一个可直接写进下一篇论文的算法框架

### 8.1 训练输入

- 训练集 \(\{(x_i,y_i)\}_{i=1}^N\)；
- 维数 \(m\)，类别数 \(K\)；
- 目标 component 数 \(r\)；
- 预处理只用训练 fold 估计；
- \(P\in\mathrm{St}(m,r)\)。

### 8.2 统计初始化

计算

\[
\hat R_c=\frac1{N_c}\sum_{i:y_i=c}x_ix_i^\top,
\quad
\hat{\bar R}=\sum_c\hat\pi_c\hat R_c,
\quad
\hat\Delta_c=\hat R_c-\hat{\bar R}.
\]

求解

\[
\max_{P^\top P=I_r}
\sum_c\hat\pi_c
\|\operatorname{diag}(P^\top\hat\Delta_cP)\|^2.
\]

可用 AJD、Riemannian optimization 或随机/PCA/MMC 多种初始化作 ablation。

### 8.3 判别微调

对 mini-batch：

1. 计算

   \[
   u_i=P^\top x_i.
   \]

2. 计算能量特征

   \[
   z_i=u_i\odot u_i.
   \]

3. 计算类别 logits

   \[
   s_i=W^\top z_i+b.
   \]

4. 最小化

   \[
   \mathcal L
   =\mathcal L_{\mathrm{CE}}
   -\alpha J_{\mathrm{contrast}}
   -\gamma J_{\mathrm{worst-pair}}
   +\beta\Omega.
   \]

5. 用 Stiefel retraction 更新 \(P\)。

### 8.4 可选的物理一致版本

若目标是保留 POVM：

\[
L_{j:}=\operatorname{softmax}(\ell_{j:}/T),
\]

\[
q_{ic}=\sum_jz_{ij}L_{jc},
\]

并使用

\[
\mathcal L_{\mathrm{NLL}}
=-\sum_i\log(q_{i,y_i}+\varepsilon).
\]

需要在实验中把它与 unrestricted energy-logit 版本分别报告，因为两者是不同模型，不应混为一个结果。

### 8.5 局限性应主动写清楚

建议下一篇论文直接列出：

1. 模型依赖二阶能量，不能识别所有高阶分布差异；
2. 纯平方模型有 \(x\leftrightarrow -x\) 不变性；
3. 共享本征基假设可能不适合强非对易类结构；
4. 全局线性基不适合高度弯曲的多流形类别；
5. 类二阶矩在 \(N\ll m\) 时需要 shrinkage；
6. 量子解释不等于量子计算加速；
7. 隐私与硬件优势需要专门方案和实验。

主动写出这些限制会使主张更可信，并直接回应 Reviewer 3。

---

## 9. ICIP Ising Clustering 草稿的数学诊断

### 9.1 当前目标的符号存在根本矛盾

草稿定义 soft assignment

\[
p_i=\operatorname{softmax}(Ax_i+b)
\]

以及 Pairwise Difference Loss

\[
\mathcal L_{\mathrm{PDL}}
=
\sum_{i,j}w_{ij}
\sum_k(p_{ik}-p_{jk})^2.
\]

文中有时说“minimize PDL”，有时又说应最大化不同样本的 assignment difference。这两者相反。

如果 \(w_{ij}\ge0\) 并最小化上式，全体

\[
p_i=p_j\quad\forall i,j
\]

就是零损失全局最优解，即 cluster collapse。

### 9.2 如果 \(w_{ij}=1\) 且最大化，目标只偏好簇大小均衡

对 one-hot assignment \(z_{ik}\in\{0,1\}\)，有

\[
\|z_i-z_j\|^2=
\begin{cases}
0,&\text{同簇},\\
2,&\text{异簇}.
\end{cases}
\]

当 \(w_{ij}=1\)：

\[
\sum_{ij}\|z_i-z_j\|^2
=2\left(N^2-\sum_kn_k^2\right),
\]

其中 \(n_k\) 是第 \(k\) 个簇的样本数。

最大化该式只会最小化 \(\sum_kn_k^2\)，即偏好大小均衡的簇；它不包含 \(x_i\) 的几何信息。

所以用 \(w_{ij}=1\) 的 Iris 结果不能证明 PDL 从数据距离中发现了簇结构。此时若结果看起来合理，来源只能是：

- 线性 softmax 参数化的归纳偏置；
- 优化路径；
- 初始化；
- 数据顺序或其他未显式描述的因素。

### 9.3 正确的图模型解释

如果 \(w_{ij}\) 表示**不相似度**，最大化

\[
\sum_{ij}w_{ij}\|z_i-z_j\|^2
\]

等价于把高不相似边尽量切开，是 weighted max-\(K\)-cut。

- \(K=2\) 可映射到 Ising spin；
- \(K>2\) 更自然是 Potts model，而不是单一二值 Ising；
- QUBO/Ising 化需要 one-hot constraint，例如

  \[
  \lambda\sum_i\left(\sum_kz_{ik}-1\right)^2.
  \]

如果 \(s_{ij}\) 表示**相似度**，更经典的是最小化图平滑项：

\[
\sum_{ij}s_{ij}\|z_i-z_j\|^2
=2\operatorname{Tr}(Z^\top L_GZ),
\]

同时加入防塌缩约束：

- \(Z^\top Z=I\)；
- balanced cluster constraint；
- global entropy maximization；
- individual entropy minimization；
- 或 normalized-cut relaxation。

这就是 spectral clustering 一类方法的标准结构。

### 9.4 当前解析推导的 eigenvalue 方向也需要复核

若目标是最大化 pairwise difference，对应二次型通常应选择较大的图 Laplacian eigenvalues；若目标是最小化相似点之间的差异，则 normalized spectral clustering 选择较小的非平凡 eigenvalues。

因此：

- “maximize difference” 与 “select smallest eigenvectors” 不能同时无条件成立；
- 必须先确定 \(w_{ij}\) 是 similarity 还是 dissimilarity；
- 必须写清是 minimize 还是 maximize；
- 只有 row-wise unit norm 约束时，不同 \(A_k\) 可能选到同一个方向；若希望得到不同簇方向，需要互相正交或其他去冗余约束。

此外，一阶 softmax 展开中的中心化项不能仅因 \(\sum_i x_i=0\) 就在所有 pairwise weighted 情形下随意删除，需要逐项检查权重和求和结构。

### 9.5 它与 ECA 的真正关系

两项工作的联系可以这样表述：

- ECA 在**特征空间**学习方向和能量算符；
- Ising/Potts clustering 在**样本图**上学习离散相互作用和分区；
- 前者研究 feature-space spectrum；
- 后者研究 sample-graph spectrum。

一个可能的统一模型是“双谱”结构：

\[
\text{feature energy }z_i=(P^\top x_i)^2
\quad\longrightarrow\quad
\text{graph/Potts assignment}.
\]

例如先由 ECA 能量构造相似图，

\[
s_{ij}
=\exp\!\left(
-\frac{\|z_i-z_j\|^2}{2\sigma^2}
\right),
\]

再做带 balance 和 entropy 约束的 spectral/Potts clustering。

但在现有 PDL 符号、塌缩和权重问题修复之前，不应宣称已经形成统一的 Ising-ECA 理论。

---

## 10. 对 TCAS-II 评审意见的研究性回应

### 10.1 编辑结论的实质

决定信不是说 ECA 没有研究价值。三位 reviewer 中：

- Reviewer 1 承认监督和无监督表现具有吸引力，但认为贡献主要是算法，没有 circuits/hardware co-design；
- Reviewer 2 认可 sparse non-overlapping features 的现象，但质疑数据规模、量子联系、复杂度、隐私和期刊范围；
- Reviewer 3 认为工作 well-motivated、theoretically sound、empirically validated，并要求补充超参数、流程和 limitations。

真正导致拒稿的是两个因素叠加：

1. **venue mismatch**：工作本质上是 ML algorithm，而不是 circuits/systems contribution；
2. **evidence gap**：实验与复杂度分析不足以支撑广泛主张。

Guest Editor 建议考虑更算法导向的 venue，这应被视为最重要的路线提示。

### 10.2 Reviewer 1：硬件与参数

合理回应不是硬加一段文字，而是做选择：

- 若投 ML venue：删除硬件优越性暗示，准确报告算法复杂度；
- 若投 circuits venue：必须给 ASIC/FPGA architecture、latency、power、area、memory traffic 和同硬件基线。

参数比较必须按第 6.2 节重新计算。不能把 kernel SVM 的 support-vector 表示参数与 ECA 的 intrinsic dimension 混用，而不同时报告实际模型存储和推理 FLOPs。

### 10.3 Reviewer 2：六个核心问题

| 评审问题 | 判断 | 需要的修复 |
|---|---|---|
| 数据太简单 | 成立 | 大规模 tabular、image embedding、synthetic stress tests |
| 量子联系弱 | 部分成立 | 用 density/PVM/POVM 严格化，删除不成立的加速暗示 |
| 参数 scaling 不清 | 成立 | raw、intrinsic、storage、FLOPs、latency 分开报告 |
| 隐私联系不足 | 成立 | 删除该主张，或新增 federated/DP threat model |
| scope 不合 | 成立 | 转算法 venue，或真正做硬件 co-design |
| downsampling 证据不足 | 成立 | repeated splits、完整 test、learning curves、CI |

### 10.4 Reviewer 3：容易修但不能单独解决拒稿

应补充：

- 完整 pseudocode；
- 超参数 selection protocol；
- sensitivity plots；
- convergence；
- limitations；
- failure cases。

但这些是必要而非充分条件。若不修正模型定位、概率一致性、参数统计和实验证据，仅补 flowchart 不会解决核心问题。

---

## 11. 下一轮实验应该怎样设计

### 11.1 合成数据：先证明“为什么 ECA 应该有效”

建议建立可控 synthetic suite：

1. **mean-only**

   各类协方差相同，仅均值不同。测试 ECA 相比 LDA 是否有不必要的复杂度。

2. **covariance-only**

   各类均值相同，协方差不同。这里线性分类器失败而 ECA/CSP/QDA 应有优势。

3. **commuting covariances**

   类协方差共享本征向量。应是 ECA 的理想场景。

4. **noncommuting covariances**

   逐步增加 commutator norm：

   \[
   \sum_{c<d}\|[R_c,R_d]\|_F.
   \]

   观察 ECA 与 full quadratic model 的差距。

5. **sign-symmetric classes**

   验证平方能量模型的优势与 \(x\leftrightarrow -x\) 不变性。

6. **identical second moments, different higher moments**

   构造 ECA 必然失败的反例，明确模型边界。

7. **\(N\ll m\)**

   比较 covariance shrinkage、低秩 \(r\)、正则化。

8. **scale shift / outliers / imbalance**

   验证 normalization、robustness 和最坏类对目标。

### 11.2 真实 tabular 数据

Reviewer 特别要求 tabular 与更难任务。建议：

- 使用 [OpenML benchmark suites](https://docs.openml.org/benchmark/) 的固定 tasks/splits；
- 使用 [TabZilla benchmark](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html) 中具有挑战性的 heterogeneous tabular datasets；
- 预先声明排除规则；
- preprocessing 完全在训练 fold 内估计；
- 不根据 test performance 选择数据集或超参数。

### 11.3 基线

至少包括：

- logistic regression；
- linear / RBF / polynomial-degree-2 SVM；
- PCA + logistic；
- LDA、shrinkage LDA、OLDA；
- MMC；
- LFDA；
- CSP 或 multiclass CSP；
- NCA、LMNN + kNN；
- QDA / regularized discriminant analysis；
- nearest-subspace / class-specific subspace；
- raw quadratic features + regularized logistic；
- random forest；
- XGBoost / LightGBM / CatBoost；
- 小型 MLP；
- random orthogonal energy features。

ECA 不能只和明显不适合的线性模型比较，也不能省略 degree-2 kernel / QDA / CSP 这些最接近其表达能力的基线。

### 11.4 评估协议

- repeated nested cross-validation；
- 所有方法相等的调参预算；
- 固定并公开 seeds；
- mean ± standard deviation / confidence interval；
- accuracy 之外报告 balanced accuracy、macro-F1、NLL、ECE；
- paired statistical tests 或跨数据集 rank analysis；
- 训练时间、推理 latency、throughput、峰值内存；
- raw parameter、intrinsic DOF、serialized bytes、FLOPs；
- learning curves：1%、2%、5%、10%、25%、50%、100% 数据；
- class imbalance 与 calibration 分析。

### 11.5 必做 ablation

1. strict orthogonal \(D=0\) vs diagonal scaling；
2. sigmoid \(L\) vs row-softmax \(L\) vs hard allocation；
3. raw projection vs square vs absolute value；
4. full \(r=m\) vs low-rank \(r\ll m\)；
5. CE only vs contrast only vs combined；
6. average margin vs worst-pair margin；
7. random / PCA / MMC / AJD initialization；
8. sparsity penalties；
9. raw / standardized / normalized inputs；
10. commuting ECA vs independent \(Q_c\)；
11. matrix exponential vs Riemannian/QR/Cayley；
12. supervised ECA vs repaired uECA。

### 11.6 解释性不应只靠热图

建议量化：

- component energy participation ratio：

  \[
  \mathrm{PR}(z)=
  \frac{(\sum_jz_j)^2}{\sum_jz_j^2};
  \]

- Hoyer sparsity / Gini；
- \(L\) 的 row entropy；
- 类别能量谱之间的距离；
- component 与类别关联在不同 seed 下的稳定性；
- joint-diagonalization residual；
- commutator norm；
- 删除 top components 后的 accuracy drop；
- 与 PCA/LDA/random basis 相同指标的比较。

只有当 ECA 的谱比基线更集中、更稳定，而且这种集中与性能或效率相关时，“discriminative sparse eigenfeatures” 才成为有证据的贡献。

---

## 12. 推荐的论文叙事

### 12.1 一句话摘要

> We learn a low-rank orthogonal basis that maximizes class-conditional energy contrast, yielding an interpretable family of commuting quadratic class operators.

### 12.2 三个主要贡献

1. 提出 class-contrast energy decomposition，并证明其与监督近似联合对角化等价；
2. 建立共享本征基二次分类器、类子空间分类和 commuting POVM 之间的统一表达；
3. 给出可扩展 Stiefel 优化和跨数据集、跨基线、带统计检验的实证验证。

### 12.3 建议标题

偏机器学习：

> **Discriminative Energy Component Analysis via Class-Conditional Joint Diagonalization**

偏理论：

> **Learning Commuting Quadratic Class Operators from Conditional Energy Contrast**

保留 ECA 品牌：

> **Eigen-Component Analysis Revisited: Class-Contrast Energy Features for Structured Quadratic Classification**

量子联系作为副线：

> **Eigen-Component Analysis: A Commuting-Measurement View of Discriminative Energy Features**

不建议继续使用：

> “Quantum-Inspired Linear Model”

因为“linear”在原输入空间中不正确，而“quantum-inspired”会使 reviewer 把注意力集中到量子新颖性和硬件实现，而不是算法核心。

### 12.4 Venue 选择

如果工作重点是：

- 新目标；
- 理论推导；
- 优化；
- tabular/vision benchmark；
- 可解释性；

那么应选择 pattern recognition / machine learning / signal processing algorithm venue。

只有在加入明确的硬件 architecture 与测量结果后，才应重新考虑 circuits-oriented venue。

Guest Editor 提到 TPAMI 是对算法方向的认可，但 TPAMI 对理论、规模、基线和实验证据要求很高；完成本报告中的关键修正和大规模实验，才有现实基础。

---

## 13. 分阶段研究路线

### Phase 1：先修正定义与实现

- [ ] 删除 `diag(D)` 或取消正交/量子概率主张；
- [ ] 在 probability-POVM 与 energy-logit 两条路线中明确选择；
- [ ] 修正参数计数；
- [ ] 把 ECA 定义为 shared-eigenbasis quadratic classifier；
- [ ] 把 uECA 的平方能量结构补回来；
- [ ] 为 Ising Clustering 修正 loss sign、collapse 和 Potts/QUBO 约束。

### Phase 2：建立理论核心

- [ ] 推导 class-energy contrast objective；
- [ ] 证明与 joint diagonalization 的等价关系；
- [ ] 给出 commuting 情形的最优性；
- [ ] 定义 noncommutativity / residual diagnostics；
- [ ] 推导二类 Helstrom/AJD baseline；
- [ ] 给出 worst-pair energy margin；
- [ ] 分析二阶矩不可分的反例。

### Phase 3：实现低秩 DECA

- [ ] \(P\in\mathrm{St}(m,r)\)；
- [ ] Riemannian/QR/Cayley optimizer；
- [ ] AJD initialization；
- [ ] shrinkage class moments；
- [ ] 稀疏与稳定性指标；
- [ ] 完整复杂度 profiler。

### Phase 4：实验

- [ ] synthetic identifiability suite；
- [ ] OpenML/TabZilla；
- [ ] 完整 baselines；
- [ ] nested CV 和 statistical tests；
- [ ] ablations；
- [ ] calibration、imbalance、small-data curves；
- [ ] 失败案例与限制。

### Phase 5：再决定扩展方向

- 如果 commuting residual 小且性能强：主打可解释结构化二次模型；
- 如果 noncommuting 数据上损失大：发展多个局部基、mixture of ECA 或 general POVM；
- 如果低秩与稀疏显著降低硬件成本：再做 FPGA/ASIC co-design；
- 如果只共享 class moments：研究 federated sufficient-statistics，但必须加入 DP/secure aggregation；
- 如果无监督目标修复成功：再连接 spectral/Potts clustering。

---

## 14. 最终判断

ECA 最有潜力的科学内核不是：

> 用量子概念把数据旋转一下，从而变成线性可分。

而是：

> 用一个共享正交基把样本的二阶方向性能量分解出来，使不同类别的条件能量谱具有最大的、可解释的差异；这些类别评分等价于一族共享本征基的二次算符。

这个内核有四个优点：

1. 它忠实对应现有算法真正计算的 \((P^\top x)^2\)；
2. 它把你的“类间差异，而不是总方差”直觉写成了严格目标；
3. 它和 joint diagonalization、CSP、class subspace、quadratic classification、POVM 都有明确关系；
4. 它产生可以被证伪、可以做 ablation、可以测复杂度的研究问题。

同时也必须接受几个硬结论：

- 正交旋转本身不创造线性可分性；
- 当前 ECA 是结构化二次模型，不是输入空间的线性模型；
- 当前代码中的 diagonal scaling 破坏正交性；
- 当前 sigmoid-\(L\) 概率解释与 CrossEntropy 实现不一致；
- 参数和效率优势尚未被现有实验普遍证明；
- small-data 不等于 privacy；
- Ising Clustering 当前 PDL 存在 sign/collapse/balance 问题；
- “最大判别差异”已有大量先例，新颖性必须落在 class-energy spectrum、commuting operators 和监督联合对角化的具体组合上。

这些修正不会削弱 ECA。相反，它们能把一个有趣的直觉变成一个边界清楚、数学扎实、可与主流方法公平比较的研究计划。

---

## 15. 主要参考文献

### ECA 原始工作

1. C. Miao and S. Ma, “Eigen Component Analysis: A Quantum Theory Incorporated Machine Learning Technique to Find Linearly Maximum Separable Components,” arXiv:2003.10199, 2020. [arXiv](https://arxiv.org/abs/2003.10199)
2. R. Chen, Y. Zhao, H. Liu, H. Xu, S. Ma, and E. Y. Lam, “Eigen-Component Analysis: A Quantum Theory Inspired Linear Model,” ISCAS 2025, DOI `10.1109/ISCAS56072.2025.11044249`. [PDF](https://www.eee.hku.hk/optima/pub/conference/2505_ISCASa.pdf)

### 判别降维与 metric learning

3. H. Li, T. Jiang, and K. Zhang, “Efficient and Robust Feature Extraction by Maximum Margin Criterion,” NeurIPS 2003. [Paper](https://proceedings.neurips.cc/paper_files/paper/2003/file/6048ff4e8cb07aa60b6777b6f7384d52-Paper.pdf)
4. M. Sugiyama, “Dimensionality Reduction of Multimodal Labeled Data by Local Fisher Discriminant Analysis,” JMLR, 2007. [Paper](https://www.jmlr.org/papers/v8/sugiyama07b.html)
5. J. Ye, “Least Squares Linear Discriminant Analysis,” ICML 2007; related orthogonal/null-space LDA formulations are surveyed in the references below.
6. K. Q. Weinberger and L. K. Saul, “Distance Metric Learning for Large Margin Nearest Neighbor Classification,” JMLR, 2009. [Paper](https://www.jmlr.org/papers/v10/weinberger09a.html)
7. J. Goldberger et al., “Neighbourhood Components Analysis,” NeurIPS 2004. [Paper](https://proceedings.neurips.cc/paper/2004/hash/42fe880812925e520249e808937738d2-Abstract.html)
8. J. P. Cunningham and Z. Ghahramani, “Linear Dimensionality Reduction: Survey, Insights, and Generalizations,” JMLR, 2015. [Paper](https://jmlr.org/papers/v16/cunningham15a.html)

### 共享本征结构、联合对角化与子空间

9. B. N. Flury, “Common Principal Components in \(k\) Groups,” JASA, 1984. [DOI](https://doi.org/10.1080/01621459.1984.10477108)
10. A. Ziehe et al., “A Fast Algorithm for Joint Diagonalization with Non-orthogonal Transformations and its Application to Blind Source Separation,” JMLR, 2004. [Paper](https://www.jmlr.org/papers/volume5/ziehe04a/ziehe04a.pdf)
11. J. Laaksonen, “Subspace Classifiers in Recognition of Handwritten Digits,” and later nearest/class-specific subspace literature; a modern analysis is available in [arXiv:1501.06060](https://arxiv.org/abs/1501.06060).
12. Common Spatial Patterns literature: CSP solves generalized variance contrast problems and is a critical baseline for two-class energy discrimination.

### 流形优化

13. N. Boumal et al., “Manopt, a Matlab Toolbox for Optimization on Manifolds,” JMLR, 2014. [Paper](https://jmlr.org/papers/v15/boumal14a.html)
14. Orthogonality-constrained optimization algorithms, JMLR 2024. [Paper](https://www.jmlr.org/papers/v25/23-0451.html)
15. J. Li et al., “Efficient Riemannian Optimization on the Stiefel Manifold via the Cayley Transform,” 2020. [arXiv](https://arxiv.org/abs/2002.01113)

### 量子测量与状态判别

16. IBM Quantum Learning, “Single systems and quantum information,” including Born-rule measurements. [Course](https://quantum.cloud.ibm.com/learning/en/courses/basics-of-quantum-information/single-systems/quantum-information)
17. IBM Quantum Learning, “General measurements,” including POVM formulations. [Course](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/general-measurements/formulations-of-measurements)
18. IBM Quantum Learning, “Density matrices.” [Course](https://quantum.cloud.ibm.com/learning/en/courses/general-formulation-of-quantum-information/density-matrices/introduction)
19. S. M. Barnett and S. Croke, “Quantum state discrimination,” Advances in Optics and Photonics, 2009. [arXiv](https://arxiv.org/abs/0810.1970)

### 图聚类、Potts 与 QUBO

20. A. Y. Ng, M. I. Jordan, and Y. Weiss, “On Spectral Clustering: Analysis and an Algorithm,” NeurIPS 2001. [Paper](https://proceedings.neurips.cc/paper_files/paper/2001/hash/801272ee79cfde7fa5960571fee36b9b-Abstract.html)
21. M. Blatt, S. Wiseman, and E. Domany, “Clustering Data through an Analogy to the Potts Model,” NeurIPS 1995. [Paper](https://proceedings.neurips.cc/paper_files/paper/1995/hash/6a2feef8ed6a9fe76d6b3f30f02150b4-Abstract.html)
22. E. G. Rieffel et al., QUBO formulations for clustering; one accessible formulation is [arXiv:1708.05753](https://arxiv.org/abs/1708.05753).

### Benchmark

23. OpenML Benchmarking Suites. [Documentation](https://docs.openml.org/benchmark/)
24. D. McElfresh et al., “When Do Neural Nets Outperform Boosted Trees on Tabular Data?” and the TabZilla benchmark, NeurIPS 2023. [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f06d5ebd4ff40b40dd97e30cee632123-Abstract-Datasets_and_Benchmarks.html)
