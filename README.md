# Smartphone Price Prediction Based on Technical Specifications

This project builds a Machine Learning pipeline to estimate the retail price (`approx_price_EUR` / `Price (EUR)`) of smartphones based on their hardware specifications, display technology, connectivity, and release timeline from the GSMArena dataset.

---

## 1. Project Directory Structure

```text
Smartphone Price Prediction/
├── data/
│   ├── phone_dataset .csv             # Raw initial dataset (8,628 devices, 40 raw columns)
│   ├── smartphones_train_clean.csv    # Imputed training set (70% - 1,337 samples)
│   ├── smartphones_val_clean.csv      # Imputed validation set (15% - 286 samples)
│   └── smartphones_test_clean.csv     # Imputed final test set (15% - 287 samples)
├── phone_dataset_cleaned.csv          # Cleaned smartphone dataset after filtering & Regex extraction (1,910 samples, 21 columns)
├── Data_cleaning.ipynb                # Step 1: Raw data cleaning & Regex specification extraction
├── analysis.ipynb                     # Step 2: Exploratory Data Analysis (EDA), Stratified Split, Imputation & Baseline Models
├── Linear_model.ipynb                 # Step 3: Feature Engineering & Linear / ElasticNet Model Optimization
├── RandomForest&XGBoost.ipynb         # Step 4: Target Encoding, Random Forest & XGBoost Tuning & Blending Ensemble
└── README.md                          # Project documentation and comprehensive results summary
```

---

## 2. Installation

Requires **Python 3.9+**. Open a Terminal / PowerShell in the project directory and install the required packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

---

## 3. How to Run the Project

Run the Jupyter Notebooks sequentially in the following pipeline order:

1. **`Data_cleaning.ipynb`**:
   * Reads the raw dataset `data/phone_dataset .csv`.
   * Filters out basic feature phones, tablets, smartwatches, non-cellular devices, and extreme luxury outliers (Vertu, price > 2,500 EUR).
   * Uses Regex to extract numeric and categorical hardware specifications, exporting `phone_dataset_cleaned.csv`.
2. **`analysis.ipynb`**:
   * Performs Exploratory Data Analysis (EDA), inspects price skewness, and visualizes relationships between hardware specifications and price.
   * Splits the dataset into **70% Train - 15% Validation - 15% Test** (using a *Stratified Split* across price segments).
   * Imputes missing values (using Median/Mode learned strictly from the Train Set to prevent Data Leakage) and exports the 3 clean CSV splits into `data/`.
   * Trains 3 Baseline models as a reference benchmark.
3. **`Linear_model.ipynb`**:
   * Iteratively experiments with Feature Engineering: physical & performance features (`PPI`, `Battery/Thickness Ratio`, `Total CPU Clock`, `Performance Index`, `Total Camera MP`) and time-relative features normalized by release year (Moore's Law: `RAM_Rel_Year`, `ROM_Rel_Year`, `Cam_Rel_Year`).
   * Applies Logarithmic transformation (`np.log1p`), `StandardScaler` normalization, and resolves high-dimensional multicollinearity using **ElasticNet** ($L_1 + L_2$) regularization.
4. **`RandomForest&XGBoost.ipynb`**:
   * Combines domain and time-relative features with **Smoothed Target Encoding** ($m=10$) for high-cardinality categorical variables (`Brand`, `Chip Brand`).
   * Evaluates Feature Importance, tunes hyperparameters via `RandomizedSearchCV`, and combines top models using a **Blending Ensemble**.

---

## 4. Detailed Summary & Model Comparison

> **Notes on Data Splits:**
> * **Validation Set (286 samples):** Used to compare and validate each Feature Engineering iteration and hyperparameter tuning step.
> * **Test Set (287 samples):** Completely held-out data, used strictly once at the end to evaluate the final generalization capability of the best models.

### 4.1. Initial Reference Benchmark (Baseline Models on Validation Set)
*Trained on the 21 original columns (with minimal preprocessing only: Train Median/Mode imputation + basic One-Hot Encoding).*

| Baseline Model | Val MAE (EUR) | Val RMSE (EUR) | Val $R^2$ Score | Preliminary Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression (Baseline)** | 58.20 € | 86.23 € | 0.6154 | Struggles to capture complex non-linear interactions |
| **XGBoost (Default)** | 51.46 € | 77.98 € | 0.6855 | Captures non-linear interactions well but is sensitive to default parameters |
| **Random Forest (Default)** | **52.32 €** | **75.88 €** | **0.7022** | Most balanced and stable baseline model prior to tuning |

---

### 4.2. Step-by-Step Linear Model Improvement Log (`Linear_model.ipynb`)
*Measured on the same **Validation Set** across each improvement iteration (with **Test Set** scores reported for the 2 optimal configurations):*

| Iteration | Technique Applied | Val MAE | Val RMSE | Val $R^2$ | Detailed Commentary |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **Step 1** | Baseline OLS (21 original columns) | 58.20 € | 86.23 € | 0.6154 | Reference starting point |
| **Step 2** | + 5 Physical & Performance Features (`PPI`, `Battery/Thickness`, `Performance Index`...) | 56.74 € | 83.92 € | 0.6357 | **Clear improvement:** Adds non-linear composite signal, boosting $R^2$ by +0.02 |
| **Step 3** | Time-Relative Features Only (Standalone Moore's Law ratios) | 59.57 € | 93.88 € | 0.5441 | Lacks detailed physical interaction features, leading to lower accuracy |
| **Step 4** | Combined Physical + Time-Relative Features (Unscaled / No Log) | 58.00 € | 91.05 € | 0.5712 | Dimensionality jumps to 98 columns, causing **severe multicollinearity** |
| **Step 5** | Full Combined + Log-Transform (`y` & `ROM`) + `StandardScaler` | 56.48 € | 91.81 € | 0.5640 | Log-transform reduces MAE, but unregularized OLS still suffers from multicollinearity |
| **Step 6** | Initial ElasticNet Trial (`alpha=0.1, l1_ratio=0.5`) | 63.69 € | 98.87 € | 0.4944 | Excessive penalty strength causes severe underfitting |
| **Step 7** | **ElasticNet Grid Tuning - MAE-Optimal Config (`alpha=0.001, l1_ratio=0.3`)** | **52.23 €** (Test: **50.63 €**) | **80.91 €** (Test: **76.12 €**) | **0.6614** (Test: **0.6721**) | **Best for MAE:** Gentle penalty stabilizes coefficients and eliminates multicollinearity |
| **Step 8** | **ElasticNet Grid Tuning - RMSE-Optimal Config (`alpha=0.001, l1_ratio=0.5`)** | **52.31 €** (Test: **51.12 €**) | **80.10 €** (Test: **77.65 €**) | **0.6681** (Test: **0.6588**) | **Best for RMSE & $R^2$:** Prevents large prediction errors on high-end devices |

---

### 4.3. Tree-Based Model Progression & Hyperparameter Tuning (`RandomForest&XGBoost.ipynb`)
*Using the 34 engineered features (Domain + Time-Relative + Smoothed Target Encoding $m=10$) and tuned via `RandomizedSearchCV`:*

| Model & Stage | Train Set | Validation Set | Commentary & Configuration |
| :--- | :---: | :---: | :--- |
| **Random Forest (Baseline FE - 34 features)** | — | MAE: 50.62 € \| RMSE: 73.46 € \| $R^2$: 0.7208 | Outperforms un-engineered baseline ($R^2$ increases from 0.7022 $\rightarrow$ 0.7208) |
| **XGBoost (Baseline FE - 34 features)** | — | MAE: 48.26 € \| RMSE: 66.90 € \| $R^2$: 0.7685 | Major leap thanks to Target Encoding and interaction features |
| **Random Forest (Filtered - 28 features)** | — | MAE: 50.31 € \| RMSE: 72.67 € \| $R^2$: 0.7268 | Pruning features with < 0.5% importance maintains comparable accuracy |
| **XGBoost (Filtered - 28 features)** | — | MAE: 48.57 € \| RMSE: 67.57 € \| $R^2$: 0.7639 | Slight drop compared to keeping all 34 features $\rightarrow$ retain all 34 features |
| **Random Forest (Tuned - 34 features)** | MAE: 22.47 € \| RMSE: 34.96 € \| $R^2$: 0.9348 | **MAE: 50.42 € \| RMSE: 72.86 € \| R^2: 0.7254** | `n_estimators=100, max_depth=15, min_samples_split=5` *(Test: MAE 48.40 € \| RMSE 69.42 € \| $R^2$ 0.7273)* |
| **XGBoost (Tuned - 34 features)** | MAE: 18.69 € \| RMSE: 25.10 € \| $R^2$: 0.9664 | **MAE: 46.19 € \| RMSE: 64.78 € \| R^2: 0.7829** | `n_estimators=300, max_depth=5, lr=0.05, subsample=0.9` *(Test: MAE 46.45 € \| RMSE 66.81 € \| $R^2$ 0.7474)* |

---

### 4.4. Final Evaluation Scores on the Held-Out Test Set (287 Independent Samples)

| Evaluated Model | Test MAE (EUR) | Test RMSE (EUR) | Test $R^2$ Score | Overall Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost (Tuned)** | **46.45 €** | **66.81 €** | **0.7474** | **Best single model across the entire project ($R^2 \approx 74.7\%$)** |
| **Ensemble Blending (0.7 XGB + 0.3 RF)** | **46.52 €** | **66.82 €** | **0.7473** | **High stability, combining bias reduction (XGB) and variance reduction (RF)** |
| **Random Forest (Tuned)** | 48.40 € | 69.42 € | 0.7273 | Strong improvement over linear models |
| **ElasticNet (MAE-Optimal)** | 50.63 € | 76.12 € | 0.6721 | Best model within the Linear Regression family |
| **ElasticNet (RMSE-Optimal)** | 51.12 € | 77.65 € | 0.6588 | Effectively controls multicollinearity in the 98-dimensional feature space |

---

## 5. Conclusions & Key Insights

1. **Progressive Performance Gains Across Stages:**
   * From the initial Baseline (Linear Regression: $\text{MAE} = 58.20\text{ €}, R^2 = 0.6154$), applying **Feature Engineering** + **Smoothed Target Encoding** + **Hyperparameter Tuning on XGBoost** reduced the Test MAE down to **46.45 €** and raised the coefficient of determination $R^2$ to **0.7474** (explaining nearly **75%** of smartphone price variance).
2. **Power of Regularization & Tree-Based Ensembles:**
   * In the linear model family, **ElasticNet** (`alpha = 0.001`) effectively resolved multicollinearity across 98 features, bringing Test MAE down to **50.63 €** ($R^2 = 0.6721$).
   * In the non-linear model family, **XGBoost** and **Random Forest** effectively leveraged time-relative hardware ratios (Moore's Law) and brand equity (`Brand_TE`), delivering a substantial leap in predictive accuracy.
3. **Top Features Driving Smartphone Value:**
   * `Diagonal Resolution (Pixels)` & `PPI` (Display sharpness), `CPU Clock Speed (GHz)` & `Performance_Index` ($\text{RAM} \times \text{Total CPU Clock}$), `ROM (GB)` (Internal storage capacity), `Cam_Rel_Year` (Camera resolution relative to release year), and `Brand_TE` (Manufacturer brand positioning).
