# 项目提案：锂电池正极设计的统计机器学习与实验设计

---

## 1. 研究问题与假设

**研究问题**：在锂电池正极设计中，**正极厚度、正极孔隙率、活性颗粒半径**三个工程可调参数如何影响放电比容量？各因子与交互作用孰为主导？能否用**高斯过程（GP）代理模型**准确刻画该响应面，并用**贝叶斯优化（BO）**在有限评估预算下高效找到最优参数组合？

**假设**（方向已按 pilot 实测修正）：

| 编号 | 假设 | 预期 | 依据 |
|---|---|---|---|
| H1 | 正极厚度对**比容量**的影响**非单调**：存在内部极大值（约 80 μm），超过后因传输限制而下降 | 内部极值，非单调 | 实测 40→200 μm：128.44 → 128.89（峰）→ 78.87 |
| H2 | 孔隙率是**主导因子**，其对比容量的影响主要来自归一化（分母 1−ε），高孔隙率提高比容量 | **强正主效应** | 实测效应 +68.1 mAh/g，是厚度的 126 倍 |
| H3 | 颗粒半径增大会降低比容量（颗粒内扩散路径变长） | 负主效应 | 实测 1→12 μm：129.53 → 125.05 |
| H4 | 厚度 × 半径存在交互（AC）：薄电极由颗粒扩散限速、厚电极由电解液传输限速 | 存在交互，但弱于孔隙率主效应 | 实测 A×C = +2.76 mAh/g |
| H5 | 响应面在**扩展区间**内高度非线性，二次 RSM 不足，GP 有效 | 二次 R² 明显低于 GP | 见 §5.2 模型对比 |
| H6 | 贝叶斯优化在有限评估预算下显著优于随机搜索 | BO 样本效率更高 | 待验证 |

## 2. 背景与意义

锂离子电池的放电比容量由电极微观结构决定，其中**正极厚度、孔隙率与活性颗粒半径**是最关键的工程可调参数。传统上这些参数的优化依赖试错实验或全参数扫描，成本高昂。本项目构建一个**可复现的计算研究框架**：

```
物理仿真器（黑箱）→ 实验设计（DOE）→ 统计机器学习（代理建模）→ 高斯过程 → 贝叶斯优化
```

**文献基础**：机器学习用于电极制造始于 Cunha et al. [A.1]；Boyce et al. [0.4] 用图像模型研究厚度×孔隙率×倍率的耦合；Hidalgo et al. [F.1a] 用 **DOE** 优化真实压延工艺；Yoo [C.2] 的 "P2D + DOE + 代理 + BO" 流程与本项目同构。

**真实数据验证**：采用公开的中试产线压延实验数据集（CC0 许可）[F.1]，将本项目方法链在真实工业数据上做可迁移性检验，这是本提案相对同类工作的一个重要补充。

**方法背景与意义**：
- **统计机器学习**：用数据拟合"参数→响应"关系的模型族，替代昂贵的试错实验与全参数扫描。本项目对比以下模型：
  - **线性回归**：假设响应是各因子的线性组合，系数直接给出每个因子的贡献大小，模型简单、可解释性最强，但无法刻画弯曲的响应面；
  - **多项式回归（二次/三次 RSM，即响应面法 Response Surface Methodology）**：在线性项之外加入平方项与交互项，能够刻画弯曲的响应面，但阶数过高时容易过拟合、外推不可靠；
  - **树模型（随机森林 / 梯度提升机 GBM，即 Gradient Boosting Machine）**：由多棵决策树投票取平均的集成方法。随机森林通过样本随机抽样与特征随机挑选训练大量互不相同的决策树，能自动捕捉非线性与交互作用、对异常值稳健；梯度提升机把上一轮预测的残差作为下一棵树的学习目标，逐轮提升精度。两者的共同短板是小样本下外推能力弱、缺乏解析的不确定性；
  - **稀疏回归（Lasso/ElasticNet）**：普通最小二乘会让所有变量的系数都不为零，即使某因子并无真实影响，噪声也会分配给它一个微小的系数，导致模型复杂、过拟合且难以解释。为此，在残差平方和之外加入一个随系数大小增长的惩罚项，迫使不重要的系数变小甚至归零：L1 惩罚（系数绝对值之和，Lasso）的约束区域带尖角，最优解常恰好落在坐标轴上，能把部分系数**精确压为 0**，实现自动变量选择；L2 惩罚（系数平方和）只能均匀缩小系数、压不到 0。ElasticNet 同时使用 L1+L2，既保留稀疏性，又在因子高度相关时更稳定。其意义在于自动识别主导因子、提高可解释性，并在小样本多特征场景下抑制过拟合。
  - **高斯过程（GP）**：见下一条，非参数贝叶斯回归，兼顾预测与不确定性量化。
- **高斯过程（GP）**：一种非参数贝叶斯回归方法，对任意输入给出预测均值 μ(x) 与预测方差 σ²(x)；既能预测响应，又能量化预测的不确定性，这是后续贝叶斯优化得以工作的基础。
- **贝叶斯优化（BO）**：以 GP 为代理模型、用采集函数（EI/UCB）序贯选择下一个评估点的全局优化方法，在"昂贵黑箱 + 有限评估预算"下具有最高的样本效率。

## 3. 因子与水平

| 因子 | 代号 | 低水平 (−) | 高水平 (+) | 单位 | 说明 |
|---|---|---|---|---|---|
| 正极厚度 | **A** | 40 | 200 | μm | 覆盖内部极值（约 80 μm）与下降段 |
| 正极孔隙率 | **B** | 0.20 | 0.60 | — | 主导因子，需更宽区间 |
| 活性颗粒半径 | **C** | 1 | 12 | μm | 覆盖更完整的扩散限制范围 |

> **关于水平间距的说明**：厚度取 40–200 μm、半径取 1–12 μm，水平间距较大是有意为之。较窄的区间会落在响应曲线的平坦段，使因子显得没有效应；扩宽后，厚度的内部极值（约 80 μm）与传输限制导致的下降段都进入设计空间，响应面的非线性才得以显现。物理上，40 μm 与 200 μm 分别对应薄电极（颗粒扩散限速）与厚电极（电解液传输限速）两种机制，1 μm 与 12 μm 覆盖了颗粒内扩散限制的完整范围，因此大间距同时放大了主效应，并使厚度×半径的交互作用可被检出。其代价是仅凭两个水平无法定位中间的极值点，故配合 LHS 空间填充与厚度扫描补齐。

**响应变量**：截止电压协议下的**放电比容量 (mAh/g)**。

- **协议**：1C 恒流放电，截止电压 **3.0 V**；
- **归一化**：

```
比容量 [mAh/g] = 放电容量 [mAh] ÷ 正极活性质量 [g]
```

其中：

```
正极活性质量 [g] = 3262 [kg/m³] × 体积 [m³] × (1 − 孔隙率) × 1000 [g/kg]
```

**单位换算说明**：`Ah/kg ≡ mAh/g` 这个等式本身是对的，但**必须保证分子分母单位一致**：

| 写法 | 结果 | 是否正确 |
|---|---|---|
| Q[mAh] ÷ m[kg] | 放大 1000 倍 | ❌ |
| Q[mAh] ÷ m[g] | 正确 | ✅ |
| Q[Ah] ÷ m[kg] | 正确 | ✅ |

**预期量级**：100–200 mAh/g（正极材料理论上限约 170–180 mAh/g）。**任何 10⁵ 量级的数值都是单位错误。**

**处理组合（2³ 全因子，8 个）**：

| 组合 | A 正极厚度 (μm) | B 孔隙率 | C 颗粒半径 (μm) |
|---|---|---|---|
| (1) | 40 | 0.20 | 1 |
| a | 200 | 0.20 | 1 |
| b | 40 | 0.60 | 1 |
| ab | 200 | 0.60 | 1 |
| c | 40 | 0.20 | 12 |
| ac | 200 | 0.20 | 12 |
| bc | 40 | 0.60 | 12 |
| abc | 200 | 0.60 | 12 |

**完整设计布局表（2 个 block，每 block 4 个组合 × 3 次重复 = 24 次）**：

| 运行 | Block | 组合 | A (μm) | B | C (μm) | 重复 |
|---|---|---|---|---|---|---|
| 1 | 1 | (1) | 40 | 0.20 | 1 | 1 |
| 2 | 1 | (1) | 40 | 0.20 | 1 | 2 |
| 3 | 1 | (1) | 40 | 0.20 | 1 | 3 |
| 4 | 1 | ab | 200 | 0.60 | 1 | 1 |
| 5 | 1 | ab | 200 | 0.60 | 1 | 2 |
| 6 | 1 | ab | 200 | 0.60 | 1 | 3 |
| 7 | 1 | ac | 200 | 0.20 | 12 | 1 |
| 8 | 1 | ac | 200 | 0.20 | 12 | 2 |
| 9 | 1 | ac | 200 | 0.20 | 12 | 3 |
| 10 | 1 | bc | 40 | 0.60 | 12 | 1 |
| 11 | 1 | bc | 40 | 0.60 | 12 | 2 |
| 12 | 1 | bc | 40 | 0.60 | 12 | 3 |
| 13 | 2 | a | 200 | 0.20 | 1 | 1 |
| 14 | 2 | a | 200 | 0.20 | 1 | 2 |
| 15 | 2 | a | 200 | 0.20 | 1 | 3 |
| 16 | 2 | b | 40 | 0.60 | 1 | 1 |
| 17 | 2 | b | 40 | 0.60 | 1 | 2 |
| 18 | 2 | b | 40 | 0.60 | 1 | 3 |
| 19 | 2 | c | 40 | 0.20 | 12 | 1 |
| 20 | 2 | c | 40 | 0.20 | 12 | 2 |
| 21 | 2 | c | 40 | 0.20 | 12 | 3 |
| 22 | 2 | abc | 200 | 0.60 | 12 | 1 |
| 23 | 2 | abc | 200 | 0.60 | 12 | 2 |
| 24 | 2 | abc | 200 | 0.60 | 12 | 3 |

> Block 1 为 ABC = − 的 4 个组合，Block 2 为 ABC = + 的 4 个组合；**ABC 与 block 混杂**。每个 block 内的运行顺序随机化（种子 20261010）。

## 4. 实验设计

| 项 | 内容 |
|---|---|
| 仿真器 | **PyBaMM 26.10.0.0**，Doyle-Fuller-Newman（DFN）模型，**Chen2020** 参数集 |
| 设计 | 2³ 全因子（24 次，**2 个 block × 4 组合 × 3 重复**）+ 拉丁超立方 LHS（20 点，GP 训练）+ CCD 轴向点（6 点，二次响应面）→ **共 50 次仿真**。其中，**拉丁超立方（LHS）** 是一种空间填充设计，把每个因子的取值范围均匀分成 n 段、每段恰好放一个采样点，使样本均匀散布在整个参数空间（而非只取全因子的"角落"），用于为高斯过程提供覆盖全空间的训练样本；**CCD 轴向点** 是中心复合设计（CCD）的一部分，沿每个因子的轴方向、在中心点之外 ±α 处取值，专门用来估计二次项（曲率），本项目取 6 点配合 2³ 因子点拟合二次响应面。 |
| **随机化** | 每个 block 内用固定随机种子（20261010）打乱运行顺序，抵御缓存/求解器状态等未知系统干扰 |
| **重复** | 仿真器是确定性的，pure error 必须来自物理或数值扰动。对每个处理组合的关键参数施加 ±1% 均匀扰动，重复 3 次，以组内方差作为 pure error 估计。重复是独立完整的仿真运行，区别于对同一次运行的重复测量。（PyBaMM DFN 使用自适应网格，粗/细网格结果完全相同，故网格分辨率不能作为重复因子） |
| **Blocking** | blocking 变量 = **求解器配置批次**（2 批：两批数值设置/初始化不同）。不同批次的数值容差与求解器状态会给响应带来系统偏移，是典型 nuisance，但其本身不是研究对象。2³ 分 2 块需要占用 1 个自由度，故将**最高阶交互 ABC 与 block 混杂**：三阶交互通常可忽略，用它换取对 nuisance 的控制是划算的。 |
| 分析工具 | Python（PyBaMM、scikit-learn）+ R（ANOVA、DiceKriging） |

> **关于误差项的说明**：确定性仿真器的 pure error 必须来自**物理或数值扰动**，不能来自网格点数。本项目采用参数 ±1% 均匀扰动作为误差来源，这既提供了非零的误差项（使 lack-of-fit 检验与 ANOVA 的 F 检验有意义），也量化了参数不确定性对响应的影响。

## 5. 分析方法（两章）

### 5.1 第一章：统计机器学习方法与训练

1. **特征工程**：三因子标准化，即减去均值、除以标准差，使各因子处于同一尺度，避免量纲大的因子主导模型；
2. **共线性诊断**：Pearson 相关矩阵（度量两两因子之间**线性**相关的强度，取值 [−1, +1]，绝对值越接近 1 表示线性相关越强；它直接对应回归中"一个因子能否被其他因子线性表示"这一问题）+ 方差膨胀因子（VIF，越大表示该变量越能被其他变量线性表示、系数越不可靠）；
3. **稀疏回归**：Lasso / ElasticNet（机制见 §2），5 折交叉验证选择正则化强度，实现自动变量选择；
4. **模型谱系对比**：5 折交叉验证（数据分 5 份轮流当验证集），以 **R² 与 RMSE** 为评估指标：线性、二次 RSM、三次、随机森林、GBM、GP-RBF、GP-Matern52；
5. **GP 超参调优**：GP 的核函数带有超参数（如长度尺度、方差、噪声水平），它们决定响应面的弯曲程度与噪声大小。超参数通过**最大化边际似然（MLE）**自动估计：边际似然是"给定超参数时观测数据出现的概率"，它同时权衡了数据拟合质量与模型复杂度（自动防止过拟合），最大化它等价于选出最可能"生成"这批数据的超参数组合，无需人工试参。由于边际似然关于超参数的函数往往非凸、存在多个局部最优，优化时采用**多次随机重启**：从多个不同的随机初始点出发各优化一次，取其中边际似然最大的一组结果，以提高找到全局最优的概率。估计出的**长度尺度**具有物理含义：长度尺度越小，表示该方向上响应变化越快、该因子越重要。
6. **不确定性校准**：留一交叉验证（LOO）下检验 2σ 预测区间覆盖率（应≈95%）与区间宽度，验证 GP 的不确定性是否"诚实"。

### 5.2 第二章：实验设计与高斯过程/贝叶斯优化

1. **主效应与交互**：基于 2³ 的 ANOVA（Type II，即在考虑其他因子的条件下逐一检验各效应）与效应估计，含交互作用的物理解释（AC 限速机制转换）；
2. **GP 代理模型**：以 50 样本拟合，评估 CV R² 与校准不确定性；
3. **贝叶斯优化**：**随机搜索（基线）vs BO-EI vs BO-UCB**（采集函数：EI 为期望改进，平衡探索与利用；UCB 为置信上界，按"均值 + κ×标准差"选点），8 个 LHS 初始点 + 20 次迭代 = 28 次评估，3 个随机种子取平均；
4. **指标**：收敛曲线上的最终最优值、"达到目标值所需评估次数"与相对随机搜索的提升幅度，直接回答"为何需要 BO 而非网格/随机搜索"。

## 6. 真实数据验证

**数据**：一个公开的中试产线压延数据集 [F.1]（CC0 许可），其设计为真实 **2×3×3 全因子**（负载 L/H × 辊温 85/120/145 °C × 目标密度 2.7/2.95/3.2 g/cm³）= 18 组 × 3 半电池 = **54 电芯**，响应为半电池容量（mAh/cm²）。

| 待检验项 | 方法 | 预期 |
|---|---|---|
| 共线性诊断 | Pearson + VIF | 真实数据可能高度共线（weight/loading/密度） |
| 稀疏回归 | Lasso / ElasticNet | 自动选出主导变量 |
| 七模型 5 折 CV | 线性/二次/三次/RF/GBM/GP-RBF/GP-Matern52 | 真实响应可能近似线性 |
| 灵敏度 | 容量 vs 各因子相关 | 与仿真结论对照 |
| BO | BO-EI/UCB vs 随机搜索 | 检验样本效率 |

## 7. 参考文献

> 分类：0=本项目直接相关；A=ML+电极制造；B=ML+电池寿命；C=贝叶斯优化+电池；D=多保真度代理模型；E=综述与BO背景；F=真实实验数据集；G=数据集目录资源。

### 0. 本项目直接相关（数据集与基础文献）

- **0.1** *Data on the effect of particle size, porosity and discharging rate on the performance of lithium-ion battery with NMC 622 cathode through numerical analysis.* Mendeley Data, 2021（CC BY 4.0），NMC622 仿真数据集（文献对照用）。
- **0.2** *A modeling approach to study the performance of Ni-rich layered oxide cathode for lithium-ion battery.* Computational Materials Science, 2021。
- **0.3** *Comprehensive Insights into the Porosity of Lithium-Ion Battery Electrodes: A Comparative Study on Positive Electrodes Based on LiNi₀.₆Mn₀.₂Co₀.₂O₂ (NMC622).* Batteries, 2021, 7(4), 70。
- **0.4** *Exploring the influence of porosity and thickness on lithium-ion battery electrodes using an image-based model.* Journal of Power Sources, 2022, 542, 231779。

### A. 机器学习 + 电池电极制造

- **A.1** *Artificial Intelligence Investigation of NMC Cathode Manufacturing Parameters Interdependencies.* Batteries & Supercaps, 2020, 3(1), 60–67。
- **A.2** *Machine Learning in Lithium-Ion Battery Cell Production: A Comprehensive Mapping Study.* Batteries & Supercaps, 2023, 6(7), e202300046。
- **A.3** *Swift Prediction of Battery Performance: Applying Machine Learning Models on Microstructural Electrode Images for Lithium-Ion Batteries.* Batteries (MDPI), 2024, 10(3), 99。
- **A.4** *Data-driven battery electrode production process modeling.* Journal of Materials Processing Technology, 2023。

### B. 机器学习 + 电池寿命 / SOH

- **B.1** *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy, 2019, 4, 383–391。
- **B.2** *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation.* Nature Communications, 2022, 13, 2261。

### C. 贝叶斯优化 + 电池

- **C.1** *Multi-Objective Bayesian Optimization of Lithium-Ion Battery Cells for Electric Vehicle Operational Scenarios.* Vehicles (MDPI), 2022。
- **C.2** *Simulation-Based Design of Lithium-Ion Battery...* PhD Thesis, Georgia Institute of Technology, 2018。
- **C.3** *Bayesian Optimization of Active Materials for Lithium-ion...* 2021。
- **C.4** *Large language model-enhanced Bayesian optimization for parameter identification of lithium-ion batteries.* Journal of Energy Storage, 2025。

### D. 多保真度 GP / 代理模型

- **D.1** *Multi-fidelity optimization via surrogate modelling.* Proceedings of the Royal Society A, 2007, 463(2088), 3251–3269。
- **D.2** *Multi-fidelity surrogate modeling for lithium-ion battery pack...* Journal of Energy Storage, 2025。
- **D.3** *Using physical property surrogate models to perform multi-fidelity optimization.* 2023。

### E. 综述与贝叶斯优化背景

- **E.1** *Artificial Intelligence Applied to Battery Research: Hype or Reality?* Chemical Reviews, 2022, 122(12), 10899–10969。
- **E.2** *Application-oriented design of machine learning paradigms for battery science.* npj Computational Materials, 2025。
- **E.3** *Taking the Human Out of the Loop: A Review of Bayesian Optimization.* Proceedings of the IEEE, 2016, 104(1), 148–175。
- **E.4** *A Tutorial on Bayesian Optimization.* arXiv:1807.02811, 2018。

### F. 真实实验数据集（可下载 / 可对照）

- **F.1** *Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing.* Mendeley Data, 2023（CC0 1.0）。
  - **F.1a** *Design of Experiments for Optimizing the Calendering Process in Li-Ion Battery Manufacturing.* Journal of Power Sources, 2023, 573, 233091。
  - **F.1b** *The Impact of Calendering Process Variables on the Impedance and Capacity Fade of Lithium-Ion Cells: An Explainable Machine Learning Approach.* Energy Technology, 2022, 10, 2200893。
- **F.2** *A limitation map of performance for porous electrodes in lithium-ion batteries.* iScience, 2021, 24(12), 103496。
- **F.3** *Machine learning optimised Li-ion battery manufacturing.* Warwick WMG, 2021。
- **F.4** *Swift Prediction of Battery Performance: Applying Machine Learning Models on Microstructural Electrode Images for Lithium-Ion Batteries.* Batteries (MDPI), 2024, 10(3), 99。
- **F.5** *"Zero" Porosity High Loading NMC622 Positive Electrodes for Li...*
- **F.6** *Comparative Evaluation of LMR-NCM and NCA Cathode...*（TUM 报告）

### G. 数据集目录 / 索引资源

- **G.1** lappemic/open-source-battery-data（GitHub）：开源电池数据集目录大全。
- **G.2** BatteryBits（Google Sheet）：公开数据集内容/质量对比表。
- **G.3** CALCE（马里兰大学）battery data：循环老化、存储、OCV、阻抗数据集。
- **G.4** NASA PCoE Battery Dataset：18650 循环老化（B0005 等）。
- **G.5** Iontech（GitHub，shiyunliu-battery）：电池数据集与开源模型资源列表。
- **G.6** BatteryML（Microsoft，已归档）：标准化电池数据集接口。
- **G.7** NLR Battery Microstructures Library：电极微观结构图像库。
- **G.8** Zenodo Battery Cell Database：电芯规格数据汇总。

---

*所有实测数值均经 PyBaMM 26.10.0.0 复现。*
