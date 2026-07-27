# 判别能量成分分析

[![研究状态](https://img.shields.io/badge/状态-理论与实验已验证-2b6cb0)](publication/main.pdf)
[![测试](https://img.shields.io/badge/测试-17%20项通过-2ea44f)](experiments/tests)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.5.1-6929c4)](experiments/scripts/run_quantum_simulation.py)
[![ISCAS 2025](https://img.shields.io/badge/ECA%20起点-ISCAS%202025-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

这个仓库把原来的 Eigen-Component Analysis（ECA）想法整理成了一个可证明、
可运行、可复现的框架：

> 学习类别判别能量成分，并把它编译成问题结构允许的最简单测量。

新的核心算法是 **Discriminative Energy Component Analysis（DECA）**：
把分类表述为“共享本征基/可交换测量”约束下的最小错误态判别。仓库现在包含：

- journal-style [论文源码](publication/main.tex)和
  [编译 PDF](publication/main.pdf)；
- 二分类解析解、可交换多分类精确性和 POVM gap 上界；
- 单调的多分类 Jacobi 算法；
- SDP oracle 与 Pretty Good Measurement（PGM）；
- 真正运行过的 Qiskit Aer PVM 和 Naimark dilation 电路；
- 17 项测试与 1,100 次 repeated-CV 外层拟合；
- 2020 原始预印本、ISCAS 2025 源文件和 Ising 聚类探索工作的研究谱系。

## 三个公式理解 DECA

把输入编码为单位态，并为每类估计一个算符：

\[
\rho_x=\phi(x)\phi(x)^\dagger,\qquad
A_c=\pi_c\,\mathbb E[\rho_x\mid y=c].
\]

对共享基 \(P=[p_1,\ldots,p_d]\)，固定基下的最优 decoder 必然可取 hard：

\[
S_{\mathrm{DECA}}(P)=
\sum_{j=1}^d\max_c p_j^\dagger A_cp_j.
\]

二分类令 \(\Delta=A_1-A_2\)，其本征基给出全局解析解：

\[
S_{\mathrm{DECA}}^\star
=\frac12(1+\|\Delta\|_1)
=S_{\mathrm{Helstrom}}^\star.
\]

这里的二分类等式就是已知 Helstrom 结果，不把它冒充新的量子判别理论。
DECA 的贡献在于约束测量表述、hard decoder 消元、多分类精确性与 gap bound、
Jacobi solver，以及严格区分：

- **PVM-DECA：**只保留本征值正负号，对应二分类最优单次物理测量；
- **Spectral-DECA：**保留本征值幅度，对应确定性二次分类或重复 shots 的
  observable expectation。

## 实验真正说明了什么

| 证据 | 结果 |
|---|---|
| 30 个随机二分类 trial | 闭式解与 SDP gap \(\le 2.0\times10^{-8}\) |
| 16 个可交换多分类 trial | DECA 与 SDP gap \(\le 1.45\times10^{-8}\) |
| 72 个非对易 trial | 理论 residual bound 零违反 |
| Qiskit Aer 电路 | 最大概率 total-variation error 为 \(0.0073\) |
| trine-state 例子 | 一般 POVM 成功率提高 \(0.04466\) |
| 经典基准 | 10 数据集、11 方法、1,100 次外层拟合 |

受控 covariance-only 数据验证了原始直觉：Spectral-DECA 达到
\(0.786\pm0.014\)，logistic regression 为 \(0.496\pm0.019\)。
公开数据结果则没有刻意美化：PGM 通常能修复一部分 PVM 多分类损失，但
RBF-SVM 和 random forest 在多数任务上仍更强。26 类 Letter Recognition
只有 17 个编码维度，实验直接暴露了可证明的 \(K>d\) PVM 容量限制。

因此本项目声称的是严格的**机制与资源—精度权衡**，不声称普遍准确率优势、
量子加速、隐私保证或硬件优势。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,test]'

.venv/bin/python -m pytest experiments/tests -q
.venv/bin/python experiments/scripts/run_theory_validation.py
.venv/bin/python experiments/scripts/run_quantum_simulation.py
```

运行完整经典实验：

```bash
.venv/bin/python experiments/scripts/run_classical_benchmarks.py \
  --folds 5 --repeats 2
.venv/bin/python experiments/scripts/analyze_classical_results.py
```

UCI 数据下载使用固定 URL 和 SHA-256 校验。下载数据缓存在 Git 忽略的
`experiments/data/`；所有 CSV、JSON、PDF 和 PNG 证据位于
[`experiments/results/`](experiments/results/)。

编译论文：

```bash
.venv/bin/python experiments/scripts/export_paper_tables.py
make -C publication
```

本地 Aer 模拟不需要 IBM Quantum 账户。

## 仓库地图

| 路径 | 内容 |
|---|---|
| [`publication/main.pdf`](publication/main.pdf) | 当前 journal-style 论文 |
| [`publication/main.tex`](publication/main.tex) | 主 LaTeX 源文件 |
| [`experiments/deca/`](experiments/deca/) | 编码、算符、solver、分类器和量子电路 |
| [`experiments/tests/`](experiments/tests/) | 理论、API 与 Qiskit 测试 |
| [`experiments/scripts/`](experiments/scripts/) | 可复现实验入口 |
| [`experiments/results/`](experiments/results/) | 原始记录、汇总和图 |
| [`references/deca_theory_and_novelty_spec.md`](references/deca_theory_and_novelty_spec.md) | 中文理论与新颖性规格 |
| [`references/eca_deep_research_analysis.md`](references/eca_deep_research_analysis.md) | 原 ECA 与 Ising 方向审计 |
| [`references/README.md`](references/README.md) | 历史材料来源与公开边界 |

## 研究谱系与作者边界

原始 ECA 论文发表于 IEEE ISCAS 2025
（[DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249)）。本仓库保留
2020 预印本和作者源文件，但不会改写历史文件来制造“新理论早已存在”的印象。

新稿暂以 `Rongzhou (Lachlan) Chen` 作为作者占位。真正投稿前，作者名单、
单位、致谢和目标期刊必须由所有人确认。

## 引用与权利

引用信息见 [`CITATION.cff`](CITATION.cff)。`experiments/` 下的原创代码使用
MIT License；历史论文、图片、手稿和模板可能有不同权利状态。复用前请阅读
[`LICENSE.md`](LICENSE.md)。
