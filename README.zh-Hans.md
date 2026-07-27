# 判别能量成分分析

[![研究状态](https://img.shields.io/badge/状态-研究路线图-8a2be2)](#项目状态)
[![ISCAS 2025](https://img.shields.io/badge/ISCAS%202025-10.1109%2FISCAS56072.2025.11044249-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![实现](https://img.shields.io/badge/代码-lachlanchen%2Feca-181717?logo=github)](https://github.com/lachlanchen/eca)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

本仓库是 **Eigen-Component Analysis（ECA）** 的研究档案与下一阶段数学路线图。它连接了 2020 年的原始想法、ISCAS 2025 发表模型、实现审计，以及更严格的新表述：

> 学习一个低秩正交基，使不同类别的条件方向性能量具有尽可能强且稳定的差异。

主要成果是
[ECA 深度研究分析](references/eca_deep_research_analysis.md)。文档明确区分了已经发表的结果、本文提出的理论、代码审计发现，以及仍需实验验证的主张。

## 一个公式理解模型

对正交基 \(P=[p_1,\ldots,p_r]\)，ECA 使用平方投影特征

\[
z_P(x)=(P^\top x)\odot(P^\top x).
\]

类别得分

\[
s_c(x)=\sum_jL_{jc}(p_j^\top x)^2
\]

等价于

\[
s_c(x)=x^\top Q_cx,\qquad
Q_c=P\operatorname{diag}(L_{:c})P^\top.
\]

因此 ECA 在能量特征上是线性的，在原输入空间中则是共享本征基的结构化二次分类器。

建议的判别目标不再最大化总体方差，而是最大化类条件能量对比：

\[
\max_{P^\top P=I}
\sum_c\pi_c
\left\|
\operatorname{diag}\!\left(P^\top(R_c-\bar R)P\right)
\right\|_2^2.
\]

它等价于对中心化类条件二阶矩做近似联合对角化。完整推导、适用边界和实验计划见深度分析。

## 仓库内容

| 路径 | 内容 | 状态 |
|---|---|---|
| [`references/eca_deep_research_analysis.md`](references/eca_deep_research_analysis.md) | 数学、物理、机器学习、实现、评审和实验分析 | 核心研究文档 |
| [`references/2003.10199v3.pdf`](references/2003.10199v3.pdf) | 原始 ECA 预印本文件 | 历史来源 |
| [`references/_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/`](references/_Rongzhou___ISCAS2025__Eigen_Component_Analysis__A_Quantum_Theory_Inspired_Linear_Model/) | ISCAS 时期 LaTeX、图，以及后续候选稿件材料 | 研究源文件档案 |
| [`references/_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/`](references/_Rongzhou___ICIP2025__Ising_Clustering__A_Democratic_Voting_Approach/) | Ising/Potts 聚类探索稿 | 未发表探索性工作 |
| [`references/1. 更 general 的数学：ECA 代表什么通用概念？.md`](references/1.%20更%20general%20的数学：ECA%20代表什么通用概念？.md) | 早期 ChatGPT 讨论，保留研究过程 | 探索笔记，不作为证据 |
| [`references/README.md`](references/README.md) | 材料来源与公开边界 | 档案说明 |

维护中的 Python 实现位于
[`lachlanchen/eca`](https://github.com/lachlanchen/eca)，本仓库不复制第二份代码。

## 关键结论

- 可逆正交变换本身不能把线性不可分数据变成线性可分；额外表达能力来自投影后的平方。
- 严格归一化且类别 effect 逐行归一的 ECA 对应可交换 POVM；硬分配对应投影测量。
- 逐元素 sigmoid 类别权重本身不能构成 categorical probability。
- 在反对称矩阵指数中加入可学习对角项通常会破坏正交性和能量守恒。
- 最有防御力的新颖性不是笼统的“最大可分性”，而是类别能量谱、共享本征基二次算符和监督联合对角化的组合。
- 当前 Ising 聚类的 pairwise-difference 目标需要明确最大化或最小化符号、加入防塌缩约束，并在多类时使用 Potts/max-\(K\)-cut 表述。

## 项目状态

1. **已发表：**ECA 论文发表于 IEEE ISCAS 2025
   （[DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249)，
   [作者公开 PDF](https://www.eee.hku.hk/optima/pub/conference/2505_ISCASa.pdf)）。
2. **已审计：**深度分析推导了真实的二次模型，并记录下一版需要修正的数学和实现不一致。
3. **待验证：**DECA 类能量对比联合对角化目标、最坏类别对 margin、低秩 Stiefel 优化和修复后的 Ising/Potts 扩展，目前都是研究提案。

本仓库不宣称量子加速、隐私保证、硬件优势或跨任务的普遍准确率优势。

## 下一步验证

建议使用可控合成数据、OpenML/TabZilla 表格任务和强基线，进行 repeated nested cross-validation、置信区间、复杂度、校准和完整 ablation。详细清单见
[分阶段研究路线](references/eca_deep_research_analysis.md#13-分阶段研究路线)。

## 引用

GitHub 会读取 [`CITATION.cff`](CITATION.cff) 并显示
**Cite this repository**。引用已发表 ECA 论文：

```bibtex
@inproceedings{chen2025eca,
  title     = {Eigen-Component Analysis: A Quantum Theory-Inspired Linear Model},
  author    = {Chen, Rongzhou and Zhao, Yaping and Liu, Hanghang and
               Xu, Haohan and Ma, Shaohua and Lam, Edmund Y.},
  booktitle = {2025 IEEE International Symposium on Circuits and Systems (ISCAS)},
  pages     = {1--5},
  year      = {2025},
  doi       = {10.1109/ISCAS56072.2025.11044249}
}
```

引用本研究档案：

```bibtex
@software{chen2026deca,
  author = {Chen, Rongzhou},
  title  = {Discriminative Energy Component Analysis: Research Archive and Roadmap},
  year   = {2026},
  url    = {https://github.com/lachlanchen/discriminative-energy-component-analysis}
}
```

欢迎提交理论修正、反例、可复现实验和范围清楚的实现。参与前请阅读
[`CONTRIBUTING.md`](CONTRIBUTING.md)。仓库包含不同来源和权利状态的论文材料，不存在覆盖全部文件的单一开源许可证；复用前请阅读
[`LICENSE.md`](LICENSE.md)。
