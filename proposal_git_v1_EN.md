# Project Proposal: Statistical Machine Learning and Design of Experiments for Lithium-Ion Cathode Design

---

## 1. Research Question and Hypotheses

**Research question**: In lithium-ion battery cathode design, how do the three engineering-tunable parameters—**cathode thickness, cathode porosity, and active particle radius**—affect the specific discharge capacity? Which factors and interactions dominate? Can a **Gaussian process (GP) surrogate model** accurately characterize this response surface, and can **Bayesian optimization (BO)** efficiently find the optimal parameter combination under a limited evaluation budget?

**Hypotheses** (directions revised according to pilot measurements):

| No. | Hypothesis | Expectation | Basis |
|---|---|---|---|
| H1 | The effect of cathode thickness on specific capacity is **non-monotonic**, with an interior maximum (about 80 μm) followed by a decline due to transport limitations | Interior extremum, non-monotonic | Measured 40→200 μm: 128.44 → 128.89 (peak) → 78.87 |
| H2 | Porosity is the **dominant factor**; its effect on specific capacity mainly comes from the normalization (denominator 1−ε), so higher porosity increases specific capacity | **Strong positive main effect** | Measured effect +68.1 mAh/g, 126 times that of thickness |
| H3 | Increasing particle radius reduces specific capacity (longer intraparticle diffusion path) | Negative main effect | Measured 1→12 μm: 129.53 → 125.05 |
| H4 | A thickness × radius interaction (AC) exists: thin electrodes are limited by particle diffusion, thick electrodes by electrolyte transport | Interaction present, weaker than the porosity main effect | Measured A×C = +2.76 mAh/g |
| H5 | Over the **extended range**, the response surface is highly nonlinear; quadratic RSM is insufficient and GP is effective | Quadratic R² clearly lower than GP | See §5.2 model comparison |
| H6 | Bayesian optimization significantly outperforms random search under a limited evaluation budget | Higher BO sample efficiency | To be verified |

## 2. Background and Significance

The specific discharge capacity of a lithium-ion battery is determined by the electrode microstructure, among which **cathode thickness, porosity, and active particle radius** are the most critical engineering-tunable parameters. Conventionally, optimizing these parameters relies on trial-and-error experiments or exhaustive parameter sweeps, which are costly. This project builds a **reproducible computational research framework**:

```
Physical simulator (black box) → Design of Experiments (DOE) → Statistical machine learning (surrogate modeling) → Gaussian process → Bayesian optimization
```

**Literature basis**: Machine learning was first applied to electrode manufacturing by Cunha et al. [A.1]; Boyce et al. [0.4] used an image-based model to study the coupling of thickness × porosity × rate; Hidalgo et al. [F.1a] used **DOE** to optimize a real calendering process; the "P2D + DOE + surrogate + BO" workflow of Yoo [C.2] is structurally identical to this project.

**Real-data validation**: We adopt a publicly available **pilot-plant calendering dataset** (CC0 license) [F.1] to test the transferability of our method chain on real industrial data, which is an important complement of this proposal relative to comparable work.

**Method background and significance**:
- **Statistical machine learning**: a family of models that fit the "parameter → response" relationship from data, replacing expensive trial-and-error experiments and exhaustive parameter sweeps. This project compares the following models:
  - **Linear regression**: assumes the response is a linear combination of the factors; coefficients directly quantify each factor's contribution. It is simple and highly interpretable but cannot capture a curved response surface;
  - **Polynomial regression (quadratic/cubic RSM, i.e., Response Surface Methodology)**: adds squared terms and interactions beyond linear terms, capturing a curved response surface, but higher orders tend to overfit and extrapolate poorly;
  - **Tree models (random forest / gradient boosting machine, GBM)**: ensemble methods that aggregate many decision trees. Random forest trains many mutually different trees via random sample bootstrapping and random feature subsampling, automatically capturing nonlinearity and interactions and being robust to outliers; gradient boosting fits each new tree to the residuals of the previous round, improving accuracy iteratively. Both share the weaknesses of weak extrapolation on small samples and the lack of analytical uncertainty;
  - **Sparse regression (Lasso/ElasticNet)**: ordinary least squares keeps all coefficients nonzero; even a factor with no real effect receives a small coefficient due to noise, making the model complex, overfit, and hard to interpret. To address this, a penalty that grows with coefficient magnitude is added to the residual sum of squares, shrinking unimportant coefficients toward or exactly to zero: the L1 penalty (sum of absolute values, Lasso) has a constraint region with sharp corners, so the optimum often falls exactly on a coordinate axis, driving some coefficients **exactly to zero** and performing automatic variable selection; the L2 penalty (sum of squares) only shrinks coefficients uniformly and cannot reach zero. ElasticNet combines L1 and L2, retaining sparsity while remaining stable when factors are highly correlated. Its value lies in automatically identifying dominant factors, improving interpretability, and suppressing overfitting in small-sample, many-feature settings.
  - **Gaussian process (GP)**: see the next item; a nonparametric Bayesian regression that provides both prediction and uncertainty quantification.
- **Gaussian process (GP)**: a nonparametric Bayesian regression method that, for any input, provides a predictive mean μ(x) and predictive variance σ²(x); it can both predict the response and quantify predictive uncertainty, which is the foundation for the subsequent Bayesian optimization.
- **Bayesian optimization (BO)**: a global optimization method that uses a GP as the surrogate and an acquisition function (EI/UCB) to sequentially select the next evaluation point; it has the highest sample efficiency for "expensive black boxes under a limited evaluation budget."

## 3. Factors and Levels

| Factor | Code | Low level (−) | High level (+) | Unit | Note |
|---|---|---|---|---|---|
| Cathode thickness | **A** | 40 | 200 | μm | Covers the interior extremum (about 80 μm) and the declining branch |
| Cathode porosity | **B** | 0.20 | 0.60 | — | Dominant factor; requires a wider range |
| Active particle radius | **C** | 1 | 12 | μm | Covers the full diffusion-limited range |

> **Note on the level spacing**: The wide spacing (40–200 μm for thickness and 1–12 μm for radius) is intentional. A narrow range falls on the plateau of the response curve, making the factor appear ineffective; after widening, both the interior extremum of thickness (about 80 μm) and the transport-limited declining branch enter the design space, so the nonlinearity of the response surface becomes visible. Physically, 40 μm and 200 μm correspond to thin-electrode (particle-diffusion-limited) and thick-electrode (electrolyte-transport-limited) regimes, respectively, while 1 μm and 12 μm cover the full range of intraparticle diffusion limitation. The wide spacing therefore amplifies the main effects and makes the thickness × radius interaction detectable. The cost is that two levels cannot locate the interior extremum, which is compensated by the LHS space-filling points and a thickness sweep.

**Response variable**: **specific discharge capacity (mAh/g)** under a cutoff-voltage protocol.

- **Protocol**: 1C constant-current discharge, cutoff voltage **3.0 V**;
- **Normalization**:

```
Specific capacity [mAh/g] = discharge capacity [mAh] ÷ cathode active mass [g]
```

where:

```
Cathode active mass [g] = 3262 [kg/m³] × volume [m³] × (1 − porosity) × 1000 [g/kg]
```

**Unit convention**: `Ah/kg ≡ mAh/g` is correct in itself, but the numerator and denominator must use consistent units:

| Expression | Result | Correct? |
|---|---|---|
| Q[mAh] ÷ m[kg] | 1000× too large | ❌ |
| Q[mAh] ÷ m[g] | Correct | ✅ |
| Q[Ah] ÷ m[kg] | Correct | ✅ |

**Expected magnitude**: 100–200 mAh/g (the theoretical upper limit of cathode materials is about 170–180 mAh/g). **Any value of order 10⁵ indicates a unit error.**

**Treatment combinations (2³ full factorial, 8 combinations)**:

| Combination | A cathode thickness (μm) | B porosity | C particle radius (μm) |
|---|---|---|---|
| (1) | 40 | 0.20 | 1 |
| a | 200 | 0.20 | 1 |
| b | 40 | 0.60 | 1 |
| ab | 200 | 0.60 | 1 |
| c | 40 | 0.20 | 12 |
| ac | 200 | 0.20 | 12 |
| bc | 40 | 0.60 | 12 |
| abc | 200 | 0.60 | 12 |

**Complete design layout table (2 blocks, 4 combinations per block × 3 replicates = 24 runs)**:

| Run | Block | Combination | A (μm) | B | C (μm) | Replicate |
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

> Block 1 contains the 4 combinations with ABC = −, and Block 2 contains the 4 combinations with ABC = +; **ABC is confounded with the block**. The run order within each block is randomized (seed 20261010).

## 4. Experimental Design

| Item | Content |
|---|---|
| Simulator | **PyBaMM 26.10.0.0**, Doyle-Fuller-Newman (DFN) model, **Chen2020** parameter set |
| Design | 2³ full factorial (24 runs, **2 blocks × 4 combinations × 3 replicates**) + Latin hypercube sampling (LHS, 20 points for GP training) + CCD axial points (6 points for the quadratic response surface) → **50 simulations in total**. Here, **LHS** is a space-filling design that divides each factor's range into n equal intervals and places exactly one sample point in each interval, spreading the samples uniformly over the parameter space (rather than only at the "corners" of the full factorial) to provide GP with training samples covering the whole space; **CCD axial points** are part of the central composite design (CCD), placed at ±α along each factor axis outside the center point and used specifically to estimate quadratic terms (curvature); this project uses 6 such points together with the 2³ factorial points to fit the quadratic response surface. |
| **Randomization** | Within each block, the run order is randomized with a fixed seed (20261010) to guard against unknown systematic effects such as caching or solver state |
| **Replication** | The simulator is deterministic, so the pure error must come from physical or numerical perturbation. For each treatment combination, a ±1% uniform perturbation is applied to key parameters, and the combination is replicated 3 times; the within-group variance estimates the pure error. Replication here means independent complete simulation runs, as distinct from repeated measurement of the same run. (PyBaMM DFN uses adaptive meshing, and coarse/fine meshes give identical results, so mesh resolution cannot serve as a replication factor.) |
| **Blocking** | The blocking variable is the **solver-configuration batch** (2 batches with different numerical settings/initialization). Differences in numerical tolerances and solver states between batches introduce systematic shifts in the response and are a typical nuisance source, although the batch itself is not an object of study. Splitting the 2³ into 2 blocks consumes 1 degree of freedom, so the **highest-order interaction ABC is confounded with the block**: the third-order interaction is usually negligible, and sacrificing it to control the nuisance is worthwhile. |
| Analysis tools | Python (PyBaMM, scikit-learn) + R (ANOVA, DiceKriging) |

> **Note on the error term**: For a deterministic simulator, the pure error must come from **physical or numerical perturbation**, not from the number of mesh points. This project uses a ±1% uniform parameter perturbation as the error source, which provides a nonzero error term (making the lack-of-fit test and the ANOVA F-test meaningful) and also quantifies the effect of parameter uncertainty on the response.

## 5. Analysis Methods (Two Chapters)

### 5.1 Chapter 1: Statistical Machine Learning Methods and Training

1. **Feature engineering**: standardize the three factors (subtract the mean and divide by the standard deviation) so that all factors are on the same scale and no factor dominates due to its unit;
2. **Collinearity diagnostics**: Pearson correlation matrix (measures the strength of the **linear** association between pairs of factors, ranging from −1 to +1; values closer to ±1 indicate stronger linear association, directly addressing the question of whether one factor can be linearly represented by the others) + variance inflation factor (VIF; larger values indicate that a variable can be more strongly represented by the other variables and that its coefficient is less reliable);
3. **Sparse regression**: Lasso / ElasticNet (mechanism in §2), with 5-fold cross-validation to select the regularization strength, performing automatic variable selection;
4. **Model-family comparison**: 5-fold cross-validation (data split into 5 folds, each serving as the validation set in turn), using **R² and RMSE** as evaluation metrics: linear, quadratic RSM, cubic, random forest, GBM, GP-RBF, GP-Matern52;
5. **GP hyperparameter tuning**: The GP kernel has hyperparameters (e.g., length scale, variance, noise level) that determine the curvature of the response surface and the noise magnitude. They are estimated automatically by **maximizing the marginal likelihood (MLE)**: the marginal likelihood is the probability of observing the data given the hyperparameters; it jointly balances goodness of fit and model complexity (automatically preventing overfitting), so maximizing it is equivalent to selecting the hyperparameters most likely to have "generated" the data, without manual tuning. Because the marginal likelihood is often non-convex in the hyperparameters and has multiple local optima, **multiple random restarts** are used: optimization starts from several different random initial points, and the result with the highest marginal likelihood is retained, increasing the chance of finding the global optimum. The estimated **length scale** has a physical meaning: a smaller length scale means the response changes faster along that direction, so the factor is more important.
6. **Uncertainty calibration**: under leave-one-out cross-validation (LOO), check the coverage of the 2σ prediction interval (should be ≈95%) and the interval width, verifying whether the GP uncertainty is "honest."

### 5.2 Chapter 2: Experimental Design and Gaussian Process / Bayesian Optimization

1. **Main effects and interactions**: ANOVA on the 2³ design (Type II, i.e., testing each effect while accounting for the other factors) and effect estimation, including the physical interpretation of the interaction (the AC rate-limiting mechanism transition);
2. **GP surrogate model**: fit with 50 samples, evaluating CV R² and calibrated uncertainty;
3. **Bayesian optimization**: **random search (baseline) vs. BO-EI vs. BO-UCB** (acquisition functions: EI is expected improvement, balancing exploration and exploitation; UCB is the upper confidence bound, selecting points by "mean + κ × standard deviation"), with 8 initial LHS points + 20 iterations = 28 evaluations, averaged over 3 random seeds;
4. **Metrics**: the final best value on the convergence curve, the "number of evaluations required to reach the target," and the improvement relative to random search, directly answering "why BO rather than grid/random search."

## 6. Real-Data Validation

**Data**: A publicly available **pilot-plant calendering dataset** [F.1] (CC0), whose design is a real **2×3×3 full factorial** (loading L/H × roll temperature 85/120/145 °C × target density 2.7/2.95/3.2 g/cm³) = 18 groups × 3 half-cells = **54 cells**, with half-cell capacity (mAh/cm²) as the response.

| Item to be tested | Method | Expectation |
|---|---|---|
| Collinearity diagnostics | Pearson + VIF | Real data may be highly collinear (weight/loading/density) |
| Sparse regression | Lasso / ElasticNet | Automatically selects dominant variables |
| Seven-model 5-fold CV | Linear/quadratic/cubic/RF/GBM/GP-RBF/GP-Matern52 | Real response may be approximately linear |
| Sensitivity | Capacity vs. each factor correlation | Compare with simulation conclusions |
| BO | BO-EI/UCB vs. random search | Test sample efficiency |

## 7. References

> Categories: 0 = directly relevant to this project; A = ML + electrode manufacturing; B = ML + battery lifetime; C = Bayesian optimization + batteries; D = multi-fidelity surrogate models; E = reviews and BO background; F = real experimental datasets; G = dataset directories and resources.

### 0. Directly Relevant to This Project (Datasets and Foundational Literature)

- **0.1** *Data on the effect of particle size, porosity and discharging rate on the performance of lithium-ion battery with NMC 622 cathode through numerical analysis.* Mendeley Data, 2021 (CC BY 4.0), NMC622 simulation dataset (for literature comparison).
- **0.2** *A modeling approach to study the performance of Ni-rich layered oxide cathode for lithium-ion battery.* Computational Materials Science, 2021.
- **0.3** *Comprehensive Insights into the Porosity of Lithium-Ion Battery Electrodes: A Comparative Study on Positive Electrodes Based on LiNi₀.₆Mn₀.₂Co₀.₂O₂ (NMC622).* Batteries, 2021, 7(4), 70.
- **0.4** *Exploring the influence of porosity and thickness on lithium-ion battery electrodes using an image-based model.* Journal of Power Sources, 2022, 542, 231779.

### A. Machine Learning + Battery Electrode Manufacturing

- **A.1** *Artificial Intelligence Investigation of NMC Cathode Manufacturing Parameters Interdependencies.* Batteries & Supercaps, 2020, 3(1), 60–67.
- **A.2** *Machine Learning in Lithium-Ion Battery Cell Production: A Comprehensive Mapping Study.* Batteries & Supercaps, 2023, 6(7), e202300046.
- **A.3** *Swift Prediction of Battery Performance: Applying Machine Learning Models on Microstructural Electrode Images for Lithium-Ion Batteries.* Batteries (MDPI), 2024, 10(3), 99.
- **A.4** *Data-driven battery electrode production process modeling.* Journal of Materials Processing Technology, 2023.

### B. Machine Learning + Battery Lifetime / SOH

- **B.1** *Data-driven prediction of battery cycle life before capacity degradation.* Nature Energy, 2019, 4, 383–391.
- **B.2** *Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation.* Nature Communications, 2022, 13, 2261.

### C. Bayesian Optimization + Batteries

- **C.1** *Multi-Objective Bayesian Optimization of Lithium-Ion Battery Cells for Electric Vehicle Operational Scenarios.* Vehicles (MDPI), 2022.
- **C.2** *Simulation-Based Design of Lithium-Ion Battery...* PhD Thesis, Georgia Institute of Technology, 2018.
- **C.3** *Bayesian Optimization of Active Materials for Lithium-ion...* 2021.
- **C.4** *Large language model-enhanced Bayesian optimization for parameter identification of lithium-ion batteries.* Journal of Energy Storage, 2025.

### D. Multi-Fidelity GP / Surrogate Models

- **D.1** *Multi-fidelity optimization via surrogate modelling.* Proceedings of the Royal Society A, 2007, 463(2088), 3251–3269.
- **D.2** *Multi-fidelity surrogate modeling for lithium-ion battery pack...* Journal of Energy Storage, 2025.
- **D.3** *Using physical property surrogate models to perform multi-fidelity optimization.* 2023.

### E. Reviews and Bayesian Optimization Background

- **E.1** *Artificial Intelligence Applied to Battery Research: Hype or Reality?* Chemical Reviews, 2022, 122(12), 10899–10969.
- **E.2** *Application-oriented design of machine learning paradigms for battery science.* npj Computational Materials, 2025.
- **E.3** *Taking the Human Out of the Loop: A Review of Bayesian Optimization.* Proceedings of the IEEE, 2016, 104(1), 148–175.
- **E.4** *A Tutorial on Bayesian Optimization.* arXiv:1807.02811, 2018.

### F. Real Experimental Datasets (Downloadable / Comparable)

- **F.1** *Data of Physical and Electrochemical Characteristics of Calendered NMC622 Electrodes and Lithium-ion Cells at Pilot-Plant Scale Battery Manufacturing.* Mendeley Data, 2023 (CC0 1.0).
  - **F.1a** *Design of Experiments for Optimizing the Calendering Process in Li-Ion Battery Manufacturing.* Journal of Power Sources, 2023, 573, 233091.
  - **F.1b** *The Impact of Calendering Process Variables on the Impedance and Capacity Fade of Lithium-Ion Cells: An Explainable Machine Learning Approach.* Energy Technology, 2022, 10, 2200893.
- **F.2** *A limitation map of performance for porous electrodes in lithium-ion batteries.* iScience, 2021, 24(12), 103496.
- **F.3** *Machine learning optimised Li-ion battery manufacturing.* Warwick WMG, 2021.
- **F.4** *Swift Prediction of Battery Performance: Applying Machine Learning Models on Microstructural Electrode Images for Lithium-Ion Batteries.* Batteries (MDPI), 2024, 10(3), 99.
- **F.5** *"Zero" Porosity High Loading NMC622 Positive Electrodes for Li...*
- **F.6** *Comparative Evaluation of LMR-NCM and NCA Cathode...* (TUM report)

### G. Dataset Directories / Index Resources

- **G.1** lappemic/open-source-battery-data (GitHub): a comprehensive directory of open-source battery datasets.
- **G.2** BatteryBits (Google Sheet): comparison of the content and quality of public datasets.
- **G.3** CALCE (University of Maryland) battery data: cycling aging, storage, OCV, and impedance datasets.
- **G.4** NASA PCoE Battery Dataset: 18650 cycling aging data (B0005 et al.).
- **G.5** Iontech (GitHub, shiyunliu-battery): a list of battery datasets and open-source model resources.
- **G.6** BatteryML (Microsoft, archived): standardized interfaces for battery datasets.
- **G.7** NLR Battery Microstructures Library: electrode microstructure image library.
- **G.8** Zenodo Battery Cell Database: a compilation of cell specification data.

---

*All measured values were reproduced with PyBaMM 26.10.0.0.*
