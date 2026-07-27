# Run 4：规范场拓扑扇区结果与算法优势审计

日期：2026-07-27
状态：精确实验已执行；41 项仓库测试通过

## 结论先行

这次结果显示了一个严格但有限定的优势：

> 当观测范围和对称性先验被明确声明时，observable contrast 不仅能找出最优
> witness，还能证明某个受限观测类根本不含类别信息。

在 \(3\times3\) toric-code / \(D(\mathbb Z_2)\) fixed point 上，所有低于码距的
1,431 个 Pauli 特征都严格看不见两个拓扑通量扇区，而算法在第一个允许的非局域
层级恢复了三条等价的三链路 Wilson 环。正确处理 nuisance logical sector 后，
学习 effect 与 Wilson-sector projector 的 Frobenius 误差为零，判别成功率从未
twirl 的 \(0.75\) 提高到 \(1\)。

这不是“算法普遍优于其他算法”的证据。给 oracle Wilson 环、Helstrom 测量或
任何同样获得 Wilson parity 特征的阈值/logistic 方法，它们同样得到 \(1\)。
因此最准确的说法是：

- 有 **observable-access advantage**：非收缩观测相对固定小于码距的局域
  观测具有严格优势；
- 有 **symmetry-prior advantage**：正确 label-preserving twirl 相对未处理
  nuisance sector 的单代表态学习具有优势；
- 有 **witness compilation / no-go certificate**：同一个谱解既恢复可实现
  witness，也量化受限代数中剩余的信息；
- 当前没有 **universal algorithmic、computational、quantum-speedup 或
  sample-complexity advantage**。

## 一、精确模型

在周期 \(L\times L\) 方格的每条边放一个 qubit，\(L=3\)，所以共有
\(n=2L^2=18\) 个 link qubits。定义

\[
A_s=\prod_{e\ni s}X_e,\qquad
B_p=\prod_{e\in\partial p}Z_e,\qquad
H=-\sum_s A_s-\sum_p B_p.
\]

所有 ground states 满足 \(A_s=B_p=+1\)。在 torus 上 ground space 是四维，
可用两个非收缩逻辑 Wilson 环 \(\bar Z_x,\bar Z_y\) 的本征值标记。实验固定
\(\bar Z_y=+1\)，比较

\[
\bar Z_x|\psi_+\rangle=+|\psi_+\rangle,\qquad
\bar Z_x|\psi_-\rangle=-|\psi_-\rangle.
\]

这里使用 toric code / quantum-double fixed point。其无电荷子空间可作
\(\mathbb Z_2\) lattice-gauge interpretation，但不把整个 toric-code Hilbert
space 等同于纯 gauge physical Hilbert space。普通 qubit partial trace 使用
extended-link/electric-center prescription；规范理论中区域代数、边界 center
和 edge modes 的选择不是唯一的。

## 二、从“最大类别差”到可访问判别距离

令 \(\Delta=\rho_+-\rho_-\)。在可访问观测代数 \(\mathcal A\) 中定义

\[
D_{\mathcal A}(\rho_+,\rho_-)
=\max_{\substack{0\preceq E\preceq I\\E\in\mathcal A}}
\operatorname{Tr}(E\Delta).
\]

当 \(\mathcal A\) 是有限维 unital \(^*\)-subalgebra，且
\(\mathcal E_{\mathcal A}\) 是保迹条件期望时，

\[
D_{\mathcal A}
=\operatorname{Tr}\!\left[
  \mathcal E_{\mathcal A}(\Delta)_+
\right]
=\frac12\left\|
  \mathcal E_{\mathcal A}(\Delta)
\right\|_1.
\]

这把原始 ECA 的“最大化 classes 之间的 difference，而非总体 variance”推进
为一个更精确的问题：不是问全空间里能否区分，而是问实际允许的观测代数里还
剩多少可判别信息。若用户提供的只是一个不封闭的 feature dictionary，而不是
\(^*\)-algebra，则不能直接套条件期望公式，应求对应 restricted SDP。

## 三、严格局域不可区分性

toric code 是 \([[18,2,3]]\) stabilizer code。对任意小于码距的支撑 \(R\)，
Knill--Laflamme / local topological order 条件给出

\[
P O_R P=c_{O_R}P,
\]

其中 \(P\) 是 code-space projector。因此任意两个 ground sectors 在该区域的
约化态相同：

\[
\rho_R^+=\rho_R^-,
\qquad
D(\rho_R^+,\rho_R^-)=0.
\]

本实验不是只抽查几个区域，而是完成两种穷举：

| 层级 | 候选数 | 非零类别差 |
|---|---:|---:|
| 1-link Pauli | 54 | 0 |
| 2-link Pauli | 1,377 | 0 |
| 合计 \(w<d\) | 1,431 | 0 |
| 3-link Pauli | 22,032 | 3 |
| 1-link RDM subsets | 18 | 0 |
| 2-link RDM subsets | 153 | 0 |
| 3-link RDM subsets | 816 | 3 |

三个非零 weight-3 Pauli 恰好是三行同伦的水平 \(ZZZ\) Wilson 环，各自满足

\[
\langle W\rangle_+-\langle W\rangle_-=2.
\]

weight 3 中共有 12 个非平凡 stabilizer-centralizer Pauli：六个 direct
\(Z\) loops 和六个 dual \(X\) loops；针对所选 \(x\)-flux label，只有上述
三个水平 \(Z\) loops 有类别差。这同时给出 code-distance certificate 和
label-specific witness certificate。

“局域观测失败”的准确含义是：固定支撑小于码距的子系统，或 weight
\(<d\) 的线性观测类，无法区分。若分别测量全部 qubits 后在经典端计算三位
parity，就已经实现了 weight-3 Wilson observable，不能再称为局域受限基线。

## 四、Wilson 环上的解析 AOC

把三个环 qubits 作为区域 \(R\)，其约化态为

\[
\rho_\pm^R=\frac{P_\pm}{2^{L-1}},\qquad
P_\pm=\frac{I\pm W}{2}.
\]

两者支撑在互相正交的 even/odd parity subspaces，因此

\[
\frac12\|\rho_+^R-\rho_-^R\|_1=1,\qquad
E^\star=P_+=\frac{I+W}{2}.
\]

数值结果是：

- loop trace distance：`1.0`；
- learned effect 与 parity projector 的 Frobenius 误差：`0.0`；
- even/odd 两个 sector 对总 trace norm 的贡献：各 `1.0`；
- 四个 ground sectors 的 orthogonality error：`0.0`；
- 所有 star/plaquette expectation 的最大误差：`0.0`。

## 五、正确和错误的 symmetry twirl

在 logical basis \(|a,b\rangle\) 中，\(a\) 是要判别的 \(x\)-flux label，
\(b\) 是测试时未知的 \(y\)-flux nuisance。每类只提供一个已知代表态
\(|0,0\rangle\) 和 \(|1,0\rangle\)。这里的“一个代表态”是已知 density/state
description；不能写成从单份未知量子态完成 tomography 或 training。

正确的 label-preserving group 是

\[
G_{\text{nuis}}=\{I,\bar X_y\},
\]

它只翻转 \(b\)。twirl 后

\[
\rho_+=\frac12\operatorname{diag}(1,1,0,0),\qquad
\rho_-=\frac12\operatorname{diag}(0,0,1,1),
\]

\[
\Delta=\frac12\operatorname{diag}(1,1,-1,-1)
=\frac12\bar Z_x.
\]

所以

\[
\operatorname{sign}(\Delta)=\bar Z_x,\qquad
E^\star=\frac{I+\bar Z_x}{2}.
\]

精确成功率如下：

| 方法 | success |
|---|---:|
| 任意固定 \(\le2\)-link / weight-\(\le2\) observer | 0.50 |
| 未 twirl、每类一个 representative 的 AOC | 0.75 |
| 只做 stabilizer twirl | 0.75 |
| 正确 label-preserving nuisance twirl + AOC | 1.00 |
| 错误 label-flipping twirl + AOC | 0.50 |
| 已知 Wilson parity threshold | 1.00 |
| full Helstrom oracle | 1.00 |

错误对照使用 \(\{I,\bar X_x\}\)，它直接翻转 label，使两类 twirled states
相同。这个负控说明“加入更多 symmetry”不一定更好；必须先判断 group action
是 nuisance-preserving 还是 label-erasing。

## 六、噪声校准

### Logical-sector mixing

令

\[
\widetilde\rho_+(p)=(1-p)\rho_+ + p\rho_-,
\qquad
\widetilde\rho_-(p)=(1-p)\rho_- + p\rho_+.
\]

则

\[
D(p)=|1-2p|,\qquad
P_{\rm succ}(p)=\frac{1+D(p)}2=1-p
\quad(0\le p\le1/2).
\]

11 个网格点上的数值最大误差为 `0.0`。这验证了 witness 在
\(p<1/2\) 时保持不变，但不是对一般 physical noise 的鲁棒性证明。

### Readout flips

若三条边各自以概率 \(r\) 独立翻转 readout，单条长度 3 Wilson parity 的成功率
为

\[
q(r)=\frac{1+(1-2r)^3}{2}.
\]

三条互不相交的同伦环做 majority vote 时，

\[
q_{\rm maj}=3q^2-2q^3.
\]

在 \(r=0.05\) 时，单环是 `0.8645`，三环多数投票是 `0.94989487775`。
这是 stabilizer-equivalent redundancy 的优势，不是 AOC 独有优势；公平的
资源比较必须计算三倍测量开销。

## 七、相对已有工作的创新边界

下列基础必须作为 prior art，而不是新发现：

1. group-invariant quantum hypothesis testing 已系统研究受对称性约束与无限制
   测量的差别；
2. symmetry-resolved entanglement 将 subsystem density matrix 按 global
   charge sectors 分块；
3. toric code、Wilson loops、局域不可区分性和拓扑量子纠错都是成熟理论；
4. ribbon-operator optimization 已用于数值发现拓扑 string operators；
5. 2026 年已经有用 fidelity/entanglement kernels 对 toric-code/extended
   toric-code states 做 unsupervised topological-order learning 的工作。

Run 4 的可辩护区别不是“第一次发现 Wilson loop”或“第一次用 ML 研究 toric
code”，而是：

> 用同一个 accessible-observable contrast formalism，同时给出局域代数的
> no-information certificate、最小非局域层级的 witness recovery、正确/错误
> symmetry twirl 对照，以及解析噪声校准。

这是一种严谨的 synthesis 和 benchmark protocol，不是一条新的 toric-code
定理。

## 八、与用户列出的四条相关基础的关系

### Group-symmetry hypothesis testing

Hiai--Mosonyi--Hayashi 研究的是 symmetry-invariant measurement 下的量子假设
检验。SAOC/AOC 的作用是把有限样本 state summary、条件期望/twirl 和可解释
witness compilation 放进同一实现。理论最优性来自已有量子判别理论。

### Symmetry-resolved entanglement

Goldstein--Sela 的 charge resolution 是 global-symmetry subsystem sectors。
Run 4 的 torus holonomy/topological-flux sector、gauge-region boundary center
和 electric-flux superselection sector 不是同一个对象。它们共享 block
decomposition 语言，但不能互相替换。

### 2026 quantum observable changepoint detection

Zecchin--Simeone--Ramdas 的 eSCD 用 classical shadows 把 measurement module
与 unknown observable detector 解耦，并控制 false-alarm average run length。
它是未来 sequential Run 5 的必须基线。当前仓库的 e-process 结果没有在相同
shots、ARL 和 held-out streams 下与 eSCD 比较，因此不能声称顺序检测优势。

### Difference-density natural orbitals

2026 DDNO 工作用 difference one-particle density matrix 分离 electron
promotion 与 orbital relaxation，并计算 transition observables。这与 ECA/AOC
共享“对差分算符做谱分解”的数学骨架，但化学可解释性来自电子结构约束、
occupation pairing 和 Slater--Condon structure，不是通用分类算法自动提供的。
仓库现有 Hückel 实验只是 controlled toy model，不是对 DDNO 的超越。

## 九、对整个仓库的优势审计

当前证据支持“匹配结构下的优势”，不支持“普遍优于其他算法”：

- Ising：AOC accuracy `0.999792`，恢复 uniform magnetization mode，
  overlap `0.99912`；RBF SVM accuracy 相同，成对差为零。
- 结构损伤：rank-1 AOC AUC `0.976796`，比共享 calibration states 的
  covariance centroid 高 `0.003526`，95% CI `[0.002527, 0.004526]`，
  并接近 oracle `0.977355`。这是目前最接近算法增益的结果，但只有 8 次重复、
  单一模拟模型，效果很小。
- 平移 vision：invariant AOC 在每类 2 个样本时 `0.999861`，但正确指定的
  Fourier-power logistic 是 `1.0`。结论是 symmetry quotient 有效，而非 AOC
  独占优势。
- 光学：达到已知 Helstrom 值 `0.95`；固定 H/V 的 `0.5` 是故意盲基线。
- robot：64 点窗口 AUC 达 `1.0`，但小窗口低于 oracle，且没有真实硬件数据和
  完整强基线。
- quantum phase、chemistry 和原 sequential experiments 目前主要是
  diagnostics，不足以形成算法性能主张。

还有两个必须修正的公平性问题：

1. 结构和 robot 实验给 AOC/covariance 方法使用了额外 calibration samples，
   不能据此宣传 sample efficiency；
2. 当前 dense 实现没有逐方法 timing/RSS scaling。循环平移的理论
   \(O(d\log d)\) 特例仍会 materialize dense matrices，尚不是实际内存优势。

## 十、下一步真正重要的问题

### 最有实际价值：surface-code syndrome drift

构造固定 marginal syndrome rates、但 correlation/string structure 改变的
在线噪声。让均值 CUSUM 天生受限，但不让强 covariance/decoder baseline
受限。比较：

- AOC/SAOC low-rank syndrome-correlation witness；
- syndrome-rate CUSUM；
- covariance changepoint detector；
- decoder log-likelihood / reweighting；
- canonical stabilizer、Wilson 和 logical-error diagnostics；
- classical-shadows eSCD；
- 相同 measured features 上的 logistic、RBF 和 sequence models。

所有方法必须使用相同总 rounds、shots、calibration、validation 和 test
预算。主要指标应是固定 false-alarm ARL 下的 detection delay、logical-error
rate，以及 witness localization。至少使用 20--50 个 noise/disorder seeds
和 held-out noise families。

只有当相对最强 non-oracle baseline 的 paired 95% CI 严格优于零，或在固定
error/delay 下显著减少 shots，才能称为新的 practical algorithmic advantage。

### 理论问题：最小可访问代数的选择

当前 \(\mathcal A\) 是人为给定的。更深的问题是联合优化

\[
\max_{\mathcal A\in\mathfrak A(C)}
D_{\mathcal A}(\rho_+,\rho_-)
-\lambda\,\mathrm{Cost}(\mathcal A),
\]

其中 \(\mathfrak A(C)\) 是满足 locality、gauge invariance、measurement depth
或 hardware connectivity 约束的观测代数族。目标不只是找 \(E^\star\)，还要找
达到指定 distinguishability 所需的最小 measurement algebra。这可能把
witness learning、observable compilation、quantum sensor design 和 error
diagnostics 统一起来。

### String theory / holography 边界

Run 4 只验证了有限 \(\mathbb Z_2\) lattice gauge / QEC testbed。它与 string
theory 共享 Wilson/'t Hooft operators、superselection sectors、operator
algebras 和 quantum-error-correction language，但没有计算 continuum gauge
theory、string worldsheet、AdS bulk reconstruction 或 CFT boundary data。

更可信的下一步是用已发表 tensor-network / holographic-code state，在明确的
boundary algebra 和 code subspace 中比较：

\[
D_{\mathcal A(R)}(\rho,\sigma),
\]

并检查 accessibility transition 是否与 entanglement-wedge reconstruction
阈值一致。这仍然是对 toy model 的数值检验，不是证明 holographic duality。

## 十一、可复现文件

- 实现：`experiments/aoc/gauge.py`
- 运行脚本：`experiments/run4/scripts/run_topological_flux.py`
- 测试：`experiments/run4/tests/test_gauge.py`
- 原始结果：`experiments/run4/results/topological_flux/`
- 机器可读摘要：
  `experiments/run4/results/topological_flux/summary.json`

命令：

```bash
.venv/bin/python -m pytest experiments/run4/tests -q
.venv/bin/python experiments/run4/scripts/run_topological_flux.py
.venv/bin/python -m pytest -q
```

本次结果目录记录的主脚本 wall time 约 13 秒；全仓 41 tests 在本机约 3.4 秒
完成。它们是小系统 exact-state proof of principle，不代表随 \(L\) 的可扩展
复杂度；dense global vector 大小为 \(2^{18}=262,144\)，代码明确阻止意外构造
大于 22 qubits 的 dense state。

## 主要参考文献

- Kitaev, *Fault-tolerant quantum computation by anyons*:
  <https://arxiv.org/abs/quant-ph/9707021>
- Bravyi, Hastings, Michalakis, *Topological quantum order: stability under
  local perturbations*: <https://arxiv.org/abs/1001.0344>
- Hiai, Mosonyi, Hayashi, *Quantum hypothesis testing with group symmetry*:
  <https://arxiv.org/abs/0904.0704>
- Goldstein, Sela, *Symmetry-resolved entanglement in many-body systems*:
  <https://arxiv.org/abs/1711.09418>
- Donnelly, *Decomposition of entanglement entropy in lattice gauge theory*:
  <https://arxiv.org/abs/1109.0036>
- Casini, Huerta, Rosabal, *Remarks on entanglement entropy for gauge fields*:
  <https://arxiv.org/abs/1312.1183>
- Bridgeman, Flammia, Poulin, *Detecting Topological Order with Ribbon
  Operators*: <https://arxiv.org/abs/1603.02275>
- Che, Gneiting, Wang, Nori, *Quantum circuit complexity and unsupervised
  machine learning of topological order*:
  <https://arxiv.org/abs/2508.04486>
- Zecchin, Simeone, Ramdas, *Universal Sequential Changepoint Detection of
  Quantum Observables via Classical Shadows*:
  <https://arxiv.org/abs/2602.11846>
- Bovill, Abou Taka, Harb, Hratchian, *Excitation/Relaxation Analysis of
  Electronic Transitions Using Difference Density Natural Orbitals*:
  <https://doi.org/10.1021/acs.jctc.5c01792>
