# 可访问观测差异研究

[![研究状态](https://img.shields.io/badge/状态-可复现工作论文-2b6cb0)](publication/)
[![测试](https://img.shields.io/badge/测试-93%20项通过-2ea44f)](experiments/)
[![研究轮次](https://img.shields.io/badge/研究轮次-5-6f42c1)](experiments/)
[![ISCAS 2025](https://img.shields.io/badge/ECA%20起点-ISCAS%202025-00629B)](https://doi.org/10.1109/ISCAS56072.2025.11044249)
[![English](https://img.shields.io/badge/README-English-blue)](README.md)

这个仓库把 ECA 的原始想法推进成了一个更一般的数学与物理框架：

> 把观测编码为正算符状态，先说明哪些测量在物理上或计算上真正可访问，
> 再学习其中期望差异最大的观测。

它不是为了替代所有分类器，而是精确解决一个同时出现在状态判别、流式变化
检测、物理模态定位、光学、化学、不变信号和多体物理中的问题。

## 核心数学

令两个经验状态之差为 \(\Delta=\rho_1-\rho_0\)。在全部有界 effect 中，

\[
\max_{0\preceq E\preceq I}\operatorname{Tr}(E\Delta)
=\operatorname{Tr}(\Delta_+)
=\frac12\|\Delta\|_1,
\qquad E^\star=\mathbf 1_{\Delta>0}.
\]

这是已知的 Helstrom/Jordan 结果，不把它冒充新的量子定理。它为 ECA 最初的
“最大化类别差异而非总体方差”给出了严格的操作含义。

run 3 进一步加入物理约束。若 \(\mathcal A\) 是可访问观测代数，
\(\mathcal E_{\mathcal A}\) 是到该代数的保迹条件期望，则

\[
\max_{\substack{E\in\mathcal A\\0\preceq E\preceq I}}
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}\!\left[
  \mathcal E_{\mathcal A}(\Delta)_+
\right].
\]

群对称性是其中一个特例：先做 twirling，再取正谱部分。表示论分块还能指出
变化由哪个 parity、frequency、charge 或其它物理扇区承载。

## 五个不可覆盖的研究轮次

| 轮次 | 问题 | 论文 |
|---|---|---|
| [run 1](experiments/run1/) | ECA 何时等价于最优可交换测量？相对一般 POVM 损失什么？ | [DECA PDF](publication/run1/main.pdf) |
| [run 2](experiments/run2/) | 最大差异观测能否逐样本累积、合并、限制秩并用于在线检测？ | [AOC PDF](publication/run2/main.pdf) |
| [run 3](experiments/run3/) | 在对称性或物理读出代数下，精确最优解是什么？哪个扇区发生变化？ | [SAOC PDF](publication/run3/main.pdf) |
| [run 4](experiments/run4/) | 能否先证明局域观测代数看不见拓扑扇区，再恢复第一个非收缩 witness？ | [Run 3+4 合并论文 PDF](publication/run4/main.pdf)；[结果与优势审计](references/run4_gauge_sector_results_and_advantage_audit.md) |
| [run 5](experiments/run5/) | syndrome 边缘不变时，相关结构能否在相同物理 cycle 预算下检测 drift 并改善解码？ | [Run 5 PDF](publication/run5/main.pdf)；[结果与优势审计](references/run5_surface_code_drift_results_and_advantage_audit.md) |

run 1 是冻结的 DECA 基线。后续轮次不会覆盖其代码、结果或论文。持续维护的
公共实现位于 [`experiments/aoc/`](experiments/aoc/)。

## 新贡献与已有理论的边界

Helstrom 判别、POVM、trace distance、群不变假设检验、条件期望、
CSP/广义特征值、kernel MMD 和 symmetry-resolved entanglement 都明确作为
已有基础引用。

本仓库真正实现并验证的是这些对象的组合：

- 可逐样本更新并精确合并的正状态统计量；
- 完整与限秩最大差异 witness；
- 带可行性审计的多分类最小错误 POVM 求解器；
- 已知独立参考状态下，可预测 witness 与 betting e-process；
- 对称性/子代数约束下的精确最优 effect；
- 可加的扇区诊断，以及 \(O(d\log d)\) 的循环平移特例；
- toric-code 局域无信息证书，以及第一个 Wilson-loop-equivalent flux witness；
- 可访问过程 no-go 证书、cycle-fair e-detector，以及受控的
  Stim/PyMatching 解码效用审计；
- 同时报告打平、失败和适用边界的跨领域实验。

完整推导见
[`additive_symmetry_observable_contrast_theory.md`](references/additive_symmetry_observable_contrast_theory.md)。

## 关键证据

| 匹配问题 | 结果 | 不美化的比较 |
|---|---:|---|
| 代数恒等式 | 解析解/SDP 最大误差 \(6.67\times10^{-9}\) | 数值精度审计 |
| 精确正负配对 Ising | AOC 准确率 \(0.9998\)，线性模型 \(0.5000\) | RBF SVM 打平，物理能量 oracle 为 1 |
| 35% 弹簧损伤、128 点窗口 | rank-1 AOC AUC \(0.9768\)，均值 logistic \(0.4985\) | covariance centroid \(0.9733\)，oracle \(0.9774\) |
| 对角/反对角偏振 | 学习 analyzer 成功率 \(0.9500\)，固定 H/V 为 \(0.5000\) | Aer 有限 shots 与解析值一致 |
| 平移干扰、每类 2 个样本 | invariant AOC \(0.999861\)，原空间方法 \(0.5000\) | 正确 Fourier-power logistic 为 \(1.0\) |
| 10-qubit TFIM 约化态 | 响应峰 \(g/J=0.9625\) | 已知热力学极限临界点为 1 |
| Hückel 差分密度 | attachment/detachment 守恒误差 \(5.55\times10^{-17}\) | 仅为受控 tight-binding 模型 |
| 模拟六轴机器人接触 | 学习 screw 重叠 \(0.99925\) | 不是硬件部署证据 |
| \(3\times3\) toric-code 通量扇区 | 1,431 个低于码距的 Pauli gap 全为 0；Wilson-loop trace distance 为 1 | 在相同 symmetry/access model 下与 Wilson、Helstrom oracle 打平 |
| \(5\times5\) 边缘保持 syndrome drift | count TV 与最大单 detector 边缘差都为 0；相关 likelihood AUC 为 \(0.5738/0.5770\) | 这是信息可访问性分离，不是 AOC 优势 |
| 同预算顺序检测 | spatial vAOC 比命名 logistic effect 慢 70.18 cycles；temporal 快 9.18 cycles 但无统计支持 | 预注册的两任务优势判定为 false |
| Stim/PyMatching 注入相关噪声，\(d=7\) | correlation-aware 解码把逻辑错误率从 \(1.526\%\) 降至 \(1.120\%\) | 已知模拟 post-channel 下相对降低 26.6%；不是 detector 或硬件结果 |

显著优势发生在预先设计的均值盲、低秩或对称干扰场景中。本项目不声称普适
准确率、量子加速、隐私保证、ASIC/FPGA 优势，也不声称已经完成真实机器人或
定量化学验证。

## 应用边界

直接应用包括光学偏振/相干测量、量子器件 drift、数据驱动序参量、结构振动
模态，以及带平移/相位干扰的 vibration、radar、sonar 和 vision 信号。

需要真实数据继续验证的方向包括机器人 force/torque 与 tactile contact、
电子差分密度和 spectroscopy、EEG/CSP 空间功率变化，以及建立在 SOAP/ACE
之上的旋转不变分子特征。

run 4 完成了这个设想的有限 fixed-point 版本：比较 toric-code flux sectors，
证明码距以下的局域不可区分性，并恢复 Wilson-loop-equivalent effect。这是
\(\mathbb Z_2\) lattice-gauge/QEC testbed，不是弦论或 holography 结果。
run 5 已执行同预算 surface-code syndrome drift 这一步：没有发现 vAOC
优于命名学习对照，但验证了相关结构能恢复边缘统计丢失的信息，并且正确选择的
post-change decoder 在受控仿真中有用。下一步需要真实硬件 syndrome stream、
在线模型选择与 decoder switching 的端到端验证。

## 快速开始

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './experiments[quantum,qec,test]'
.venv/bin/python -m pytest -q
```

运行 run 2：

```bash
.venv/bin/python experiments/run2/scripts/run_algebraic_validation.py
.venv/bin/python experiments/run2/scripts/run_ising_order.py
.venv/bin/python experiments/run2/scripts/run_structural_monitoring.py
.venv/bin/python experiments/run2/scripts/run_optical_quantum.py
```

运行 run 3：

```bash
.venv/bin/python experiments/run3/scripts/run_translation_vision.py
.venv/bin/python experiments/run3/scripts/run_quantum_phase.py
.venv/bin/python experiments/run3/scripts/run_huckel_difference.py
.venv/bin/python experiments/run3/scripts/run_robot_contact.py
```

运行 run 4 的精确规范扇区实验：

```bash
.venv/bin/python experiments/run4/scripts/run_topological_flux.py
```

运行 run 5 的冻结实验：

```bash
.venv/bin/python experiments/run5/scripts/run_identifiability_certificate.py
.venv/bin/python experiments/run5/scripts/run_syndrome_drift.py \
  --config experiments/run5/configs/paper.json
.venv/bin/python experiments/run5/scripts/run_offline_diagnostic_audit.py \
  --config experiments/run5/configs/offline_diagnostic_locked.json
.venv/bin/python experiments/run5/scripts/run_shadow_measurement_audit.py
.venv/bin/python experiments/run5/scripts/run_circuit_level_validation.py
```

构建全部论文：

```bash
make -C publication
```

本地 Aer 不需要 IBM Quantum 账户。每个新结果目录都包含原始 CSV/JSON 与
manifest，其中记录命令、依赖、Git 状态、运行时间和输出哈希。

## 仓库地图

| 路径 | 内容 |
|---|---|
| [`experiments/aoc/`](experiments/aoc/) | 加性、多分类、流式、对称、物理、化学和量子基础实现 |
| [`experiments/run1/`](experiments/run1/) | 冻结的 DECA 代码、测试、脚本和证据 |
| [`experiments/run2/`](experiments/run2/) | 加性/流式观测差异验证 |
| [`experiments/run3/`](experiments/run3/) | 对称分辨与跨领域验证 |
| [`experiments/run4/`](experiments/run4/) | 精确局域不可区分性与拓扑通量验证 |
| [`experiments/run5/`](experiments/run5/) | cycle-fair syndrome drift、测量与解码验证 |
| [`publication/`](publication/) | run 1–5 的五份独立论文源码与编译 PDF |
| [`references/`](references/) | ECA/Ising 原始材料、审稿意见、研究计划和理论分析 |

## 研究谱系、引用与权利

原始 ECA 论文发表于 IEEE ISCAS 2025
（[DOI](https://doi.org/10.1109/ISCAS56072.2025.11044249)）。2020 预印本、
ISCAS 源文件、Ising 聚类探索稿和早期讨论保存在 [`references/`](references/)
中作为 provenance，不会改写历史来制造“后来理论早已存在”的印象。

五份工作稿暂以 `Rongzhou (Lachlan) Chen` 作为仓库作者占位。正式投稿前，
作者、单位、致谢和期刊必须由人类合作者确认。

GitHub 使用 [`CITATION.cff`](CITATION.cff) 生成引用界面：

```bibtex
@software{chen2026observablecontrast,
  author  = {Chen, Rongzhou},
  title   = {Observable Contrast Research:
             From Eigen-Components to Additive and
             Symmetry-Resolved Physical Witnesses},
  year    = {2026},
  version = {5.0.0},
  url     = {https://github.com/lachlanchen/discriminative-energy-component-analysis}
}
```

[`experiments/`](experiments/) 下的原创代码采用 MIT License；历史论文、
图片、模板和出版物可能有不同权利。详见 [`LICENSE.md`](LICENSE.md)。
