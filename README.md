<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Data%20Science%20Portfolio&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=10%20End-to-End%20Tasks%20%C2%B7%20Python%20%C2%B7%20Pandas%20%C2%B7%20Streamlit&descAlignY=58&descSize=16" width="100%"/>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557c?style=for-the-badge&logo=python&logoColor=white)](https://matplotlib.org)
[![Seaborn](https://img.shields.io/badge/Seaborn-Stats%20Viz-76c0c1?style=for-the-badge&logo=python&logoColor=white)](https://seaborn.pydata.org)

<br/>

[![Tasks](https://img.shields.io/badge/Tasks%20Completed-10%20%2F%2010-00C853?style=flat-square)](.) [![Notebooks](https://img.shields.io/badge/Notebooks-10-blue?style=flat-square&logo=jupyter)](.) [![Cells](https://img.shields.io/badge/Total%20Cells-450%2B-orange?style=flat-square)](.) [![Lines](https://img.shields.io/badge/Lines%20of%20Code-3%2C500%2B-purple?style=flat-square)](.) [![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

<br/>

> **A comprehensive, end-to-end Data Science portfolio** covering statistical analysis, data cleaning, EDA, regression, hypothesis testing, visualisation, and interactive Streamlit deployment.

<br/>

[🚀 **Live Demo**](#-live-demo) · [📓 **Notebooks**](#-notebooks) · [🛠️ **Tech Stack**](#-tech-stack) · [📊 **Task Overview**](#-task-overview) · [⚙️ **Installation**](#-installation)

</div>

---

## 🗺️ Portfolio Navigation

<div align="center">

| | Task | Category | Key Techniques | Status |
|:---:|:---|:---:|:---|:---:|
| 📊 | [Task 1 — Statistical Analysis](#-task-1--statistical-analysis) | Statistics | ANOVA · Normality Tests · Descriptive Stats | ✅ |
| 🧹 | [Task 2 — Data Cleaning](#-task-2--data-cleaning) | Engineering | Imputation · Dedup · Type Conversion | ✅ |
| 🔭 | [Task 3 — Exploratory EDA](#-task-3--exploratory-eda) | Analysis | Univariate · Bivariate · Correlation | ✅ |
| 📈 | [Task 4 — Linear Regression](#-task-4--linear-regression) | ML | OLS · Ridge · Lasso · Diagnostics | ✅ |
| 🎨 | [Task 5 — Seaborn Visualisation](#-task-5--seaborn-visualisation) | Visualisation | 20+ chart types · KDE · Facets | ✅ |
| 🧪 | [Task 6 — Hypothesis Testing](#-task-6--hypothesis-testing) | Statistics | t-test · ANOVA · Chi-Square · Power | ✅ |
| 🖼️ | [Task 7 — Matplotlib Visualisation](#-task-7--matplotlib-visualisation) | Visualisation | GridSpec · 30+ chart types · Dashboards | ✅ |
| 🔗 | [Task 8 — Correlation Matrix](#-task-8--correlation-matrix) | Analysis | Pearson · Spearman · Clustering | ✅ |
| 🧼 | [Task 9 — Pandas Data Cleaning](#-task-9--pandas-data-cleaning-advanced) | Engineering | 14 Issues · 40+ Methods · Validation | ✅ |
| 🐼 | [Task 10 — Data Manipulation](#-task-10--data-manipulation) | Engineering | Filter · GroupBy · Merge · Reshape | ✅ |

</div>

---

## 🚀 Live Demo

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)

```
streamlit run app.py
```

> One unified Streamlit app — all 10 tasks with interactive controls, live charts, and real-time parameter tuning.

</div>

---

## 📊 Task 1 — Statistical Analysis

![Dataset](https://img.shields.io/badge/Dataset-Iris%20150×4-blue?style=flat-square) ![Tests](https://img.shields.io/badge/Tests-ANOVA·Shapiro--Wilk·Pearson-orange?style=flat-square) ![Charts](https://img.shields.io/badge/Charts-6-green?style=flat-square)

**Objective:** Perform comprehensive statistical analysis on the Iris dataset to understand feature distributions, species differences, and inter-feature relationships.

<details>
<summary><b>📋 What is covered</b></summary>
<br/>

| Step | Description | Method |
|------|-------------|--------|
| Descriptive stats | Mean, median, std, quartiles per species | `.describe()`, `.groupby().agg()` |
| Normality testing | Shapiro-Wilk on each feature × species | `scipy.stats.shapiro()` |
| One-Way ANOVA | Species differences for all 4 features | `scipy.stats.f_oneway()` |
| Correlation | Pearson pairwise between features | `.corr()`, `seaborn.heatmap()` |
| Visualisation | Histogram · Box · Violin · Scatter · Heatmap | `matplotlib`, `seaborn` |

</details>

<details>
<summary><b>📈 Key Findings</b></summary>
<br/>

- All 4 features show **statistically significant** differences across species (ANOVA p < 0.0001)
- **Petal length** is the most discriminative feature (F ≈ 1180)
- Strong positive correlation: petal length vs petal width (r ≈ 0.96)
- Setosa is clearly linearly separable from Versicolor and Virginica

</details>

```python
from scipy.stats import f_oneway

for feature in iris.select_dtypes("number").columns:
    groups = [iris[iris["species"] == sp][feature] for sp in iris["species"].unique()]
    F, p   = f_oneway(*groups)
    print(f"{feature:15}: F={F:.2f}  p={p:.2e}  {chr(10003) if p < 0.05 else chr(10007)}")
```

---

## 🧹 Task 2 — Data Cleaning

![Issues](https://img.shields.io/badge/Issues%20Fixed-12-red?style=flat-square) ![Rows](https://img.shields.io/badge/Rows-1200-blue?style=flat-square) ![Clean](https://img.shields.io/badge/Completeness-99.8%25-green?style=flat-square)

**Objective:** Transform a messy retail dataset into a clean, validated DataFrame with a full audit trail.

<details>
<summary><b>📋 Cleaning Pipeline</b></summary>
<br/>

```
RAW DATA (1,225 rows)
    |
    +-- 1. Column names -> snake_case
    +-- 2. "$31,420" -> 31420.0  (pd.to_numeric)
    +-- 3. Mixed date formats -> datetime64
    +-- 4. Yes/No/Y/N -> 0/1 binary
    +-- 5. Missing values: 7 strategies
    +-- 6. Exact duplicates: 30 removed
    +-- 7. Near-duplicates: 18 removed
    +-- 8. Invalid Age (<18, >100): median imputation
    +-- 9. Invalid Satisfaction: mode imputation
    +-- 10. Email validation: regex + flag
    +-- 11. IQR Winsorisation: outliers
    +-- 12. Ordered Categorical: segment

CLEAN DATA (1,177 rows) [OK]
```

</details>

<details>
<summary><b>📈 Before vs After</b></summary>

| Metric | Before | After |
|--------|--------|-------|
| Rows | 1,225 | 1,177 |
| Missing cells | 412 | 0 |
| Duplicate rows | 48 | 0 |
| Unique gender values | 15+ | 4 |
| `annual_income` dtype | `object` | `float64` |
| Completeness | 91.2% | 100% |

</details>

```python
# Group-aware missing value imputation
df["annual_income"] = (
    df.groupby("segment")["annual_income"]
      .transform(lambda x: x.fillna(x.median()))
)
```

---

## 🔭 Task 3 — Exploratory EDA

![Variables](https://img.shields.io/badge/Variables-20-blue?style=flat-square) ![Charts](https://img.shields.io/badge/Chart%20Types-12-orange?style=flat-square) ![Insights](https://img.shields.io/badge/Insights-15%2B-green?style=flat-square)

**Objective:** Systematic exploration of the retail dataset — distributions, relationships, and business patterns.

<details>
<summary><b>📋 EDA Framework</b></summary>

| Phase | Analyses | Output |
|-------|----------|--------|
| **Univariate** | Histograms, KDE, box plots, Q-Q plots | Shape, skewness, outliers |
| **Categorical** | Value counts, bar charts, percentages | Cardinality, imbalance |
| **Bivariate** | Scatter, grouped box, violin, cross-tabs | Relationships |
| **Temporal** | Line charts, resample, cohort analysis | Trends, seasonality |
| **Correlation** | Heatmap, pairplot, Pearson/Spearman | Feature relationships |

</details>

```python
monthly_metrics = (
    df.set_index("join_date")
      .resample("ME")
      .agg(new_customers=("customer_id", "count"),
           avg_spend=("annual_spend", "mean"),
           total_revenue=("revenue", "sum"))
)
```

---

## 📈 Task 4 — Linear Regression

![Models](https://img.shields.io/badge/Models-OLS·Ridge·Lasso-blue?style=flat-square) ![CV](https://img.shields.io/badge/CV-5--Fold-orange?style=flat-square) ![Metrics](https://img.shields.io/badge/R²=0.684-green?style=flat-square)

**Objective:** Build, evaluate, and diagnose regression models predicting annual spend.

<details>
<summary><b>📋 Model Results</b></summary>

| Model | Test R² | RMSE | CV R² |
|-------|---------|------|-------|
| Simple OLS | 0.412 | $4,821 | 0.408 |
| Multiple OLS (5 features) | 0.681 | $3,567 | 0.674 |
| Ridge (alpha=1.0) | 0.684 | $3,549 | 0.679 |
| Lasso (alpha=0.5) | 0.682 | $3,558 | 0.676 |

</details>

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

ridge     = Ridge(alpha=1.0)
cv_scores = cross_val_score(ridge, X_scaled, y, cv=5, scoring="r2")
print(f"CV R² = {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
```

---

## 🎨 Task 5 — Seaborn Visualisation

![Charts](https://img.shields.io/badge/Chart%20Types-20%2B-blue?style=flat-square) ![Panels](https://img.shields.io/badge/Panels-18-orange?style=flat-square)

**Objective:** Master Seaborn's statistical visualisation API for publication-quality charts.

<details>
<summary><b>📋 Chart Library</b></summary>

| Category | Functions |
|----------|-----------|
| **Distribution** | `histplot`, `kdeplot`, `ecdfplot`, `rugplot` |
| **Categorical** | `boxplot`, `violinplot`, `stripplot`, `barplot`, `countplot` |
| **Relational** | `scatterplot`, `lineplot`, `regplot`, `residplot` |
| **Matrix** | `heatmap`, `clustermap` |
| **Multi-plot** | `pairplot`, `FacetGrid`, `catplot`, `relplot` |

</details>

```python
g = sns.FacetGrid(df, col="segment",
                  col_order=["Bronze","Silver","Gold","Platinum"],
                  height=4, aspect=0.9)
g.map_dataframe(sns.histplot, x="annual_spend", kde=True)
g.set_titles(col_template="{col_name}")
```

---

## 🧪 Task 6 — Hypothesis Testing

![Tests](https://img.shields.io/badge/Tests-8%20types-blue?style=flat-square) ![Effect](https://img.shields.io/badge/Effect%20Sizes-Cohen's%20d·η²·r-orange?style=flat-square)

**Objective:** Apply rigorous statistical testing to validate business hypotheses.

<details>
<summary><b>📋 Tests & Results</b></summary>

| Hypothesis | Test | p-value | Decision |
|------------|------|---------|----------|
| Spend differs across segments | One-Way ANOVA | <0.0001 | ✅ Reject H₀ |
| Platinum spend > Bronze | Welch t-test | <0.0001 | ✅ Reject H₀ |
| Segment is independent of Channel | Chi-Square | 0.109 | ❌ Fail to reject |
| Discount users spend more | Welch t-test | 0.0001 | ✅ Reject H₀ |
| Spend non-normal by segment | Kruskal-Wallis | <0.0001 | ✅ Reject H₀ |

</details>

```python
from scipy.stats import ttest_ind

bronze   = df[df["segment"] == "Bronze"]["annual_spend"]
platinum = df[df["segment"] == "Platinum"]["annual_spend"]
t, p = ttest_ind(bronze, platinum, equal_var=False)  # Welch
cohens_d = (platinum.mean() - bronze.mean()) / \
    ((bronze.std()**2 + platinum.std()**2) / 2) ** 0.5
```

---

## 🖼️ Task 7 — Matplotlib Visualisation

![Charts](https://img.shields.io/badge/Chart%20Types-30%2B-blue?style=flat-square) ![GridSpec](https://img.shields.io/badge/GridSpec-Multi--panel-orange?style=flat-square)

**Objective:** Build publication-quality, custom visualisations using Matplotlib's low-level API.

```python
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(20, 14))
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)

ax1 = fig.add_subplot(gs[0, :2])   # Wide revenue trend
ax2 = fig.add_subplot(gs[0, 2])    # Segment pie
ax3 = fig.add_subplot(gs[1, :])    # Full-width scatter
ax4 = fig.add_subplot(gs[2, :3])   # Grouped bar
```

---

## 🔗 Task 8 — Correlation Matrix

![Methods](https://img.shields.io/badge/Methods-Pearson·Spearman·Kendall-blue?style=flat-square) ![Features](https://img.shields.io/badge/Features-9%20numeric-orange?style=flat-square)

**Objective:** Compute, visualise, and interpret multi-method correlation matrices with significance testing.

<details>
<summary><b>📋 Analysis Suite</b></summary>

| Analysis | Method | Output |
|----------|--------|--------|
| Pearson | Linear relationship | Standard heatmap |
| Spearman | Monotonic (non-parametric) | Rank heatmap |
| Kendall's tau | Concordance pairs | Concordance matrix |
| Significance | Pairwise p-values | -log10(p) heatmap |
| Clustered | Ward linkage | Dendrogram + heatmap |

</details>

```python
from scipy.stats import pearsonr
import itertools

pvals = pd.DataFrame(index=num_cols, columns=num_cols, dtype=float)
for i, j in itertools.combinations(num_cols, 2):
    _, p = pearsonr(df[i], df[j])
    pvals.loc[i, j] = pvals.loc[j, i] = p

neg_log_p = -np.log10(pvals.fillna(1))
```

---

## 🧼 Task 9 — Pandas Data Cleaning (Advanced)

![Issues](https://img.shields.io/badge/Issues-14%20categories-red?style=flat-square) ![Methods](https://img.shields.io/badge/Pandas%20Methods-40%2B-blue?style=flat-square) ![Checks](https://img.shields.io/badge/Validation%20Checks-20-green?style=flat-square)

**Objective:** Apply every major Pandas cleaning technique — 14 injected quality issues, 40+ methods, 20 validation assertions.

<details>
<summary><b>📋 14 Data Quality Issues</b></summary>

| # | Category | Example | Fix |
|---|----------|---------|-----|
| 1 | Column whitespace | `"Annual Income "` | `.str.strip().str.lower()` |
| 2 | Missing values 5-12% | `NaN` in 6 columns | Segment-median / mode / Unknown |
| 3 | Exact duplicates | 30 rows | `.drop_duplicates()` |
| 4 | Near-duplicates | 18 rows (tweaked) | `subset=["name","email"]` |
| 5 | Dollar-string | `"$31,420"` | `.str.replace()` + `pd.to_numeric()` |
| 6 | Comma-formatted | `"18,320.50"` | `.str.replace(",","")` |
| 7 | Mixed dates | `"01/15/2022"`, `"Jan 15, 2022"` | `pd.to_datetime(infer=True)` |
| 8 | Gender typos | `"male"`, `"m"`, `"mal"` | `.str.lower().map(norm)` |
| 9 | Segment typos | `"bronze"`, `"Bronz"` | `.str.lower().map(norm)` |
| 10 | Invalid Age | `-5`, `150`, `999` | Range mask + median |
| 11 | Invalid Satisfaction | `0`, `6`, `7` | Range mask + mode |
| 12 | Boolean strings | `"Yes"`, `"TRUE"`, `"N"` | `.map(bool_map)` |
| 13 | Statistical outliers | Income $400k-$1.5M | IQR Winsorisation |
| 14 | Bad emails | `"notanemail"`, `"user@"` | Regex + flag |

</details>

```python
# Ordered Categorical enables segment-based comparisons
df["segment"] = pd.Categorical(
    df["segment"],
    categories=["Bronze", "Silver", "Gold", "Platinum"],
    ordered=True
)

# Boolean condition using ordered categorical
premium = df[(df["segment"] > "Silver") & (df["annual_spend"] > df["annual_spend"].median())]
```

---

## 🐼 Task 10 — Data Manipulation

![Techniques](https://img.shields.io/badge/Techniques-15%20categories-blue?style=flat-square) ![Questions](https://img.shields.io/badge/Business%20Questions-10%20answered-orange?style=flat-square) ![Assertions](https://img.shields.io/badge/Assertions-20%20passed-green?style=flat-square)

**Objective:** Apply every major Pandas manipulation technique to answer 10 real business questions.

<details>
<summary><b>📋 Technique Reference</b></summary>

| Category | Methods |
|----------|---------|
| **Selecting** | `.loc[]`, `.iloc[]`, `.filter()`, `.query()`, `.nlargest()` |
| **Aggregation** | `.groupby().agg()`, `.pivot_table()`, `pd.crosstab()` |
| **Transform** | `.transform()`, `.expanding()`, `.cumsum()` |
| **Calculations** | `.assign()`, `.apply()`, `.map()`, `np.log1p()` |
| **DateTime** | `.dt.year`, `.dt.day_name()`, `.resample()` |
| **Merging** | `.merge(how="left", indicator=True)`, `.concat()` |
| **Reshape** | `.melt()`, `.pivot()`, `.stack()`, `.unstack()` |
| **Windows** | `.rolling(window=)`, `.expanding()`, `.shift()` |
| **Encode** | `pd.get_dummies()`, `LabelEncoder`, ordinal map |

</details>

<details>
<summary><b>💼 10 Business Questions</b></summary>

| # | Question | Finding |
|---|----------|---------|
| Q1 | Which segment generates highest revenue? | Platinum: highest per-customer |
| Q2 | Month-over-month spend trend? | +3.2% avg MoM growth |
| Q3 | Best rated category per region? | Electronics leads all regions |
| Q4 | Do discount users spend more? | +17.4% lift |
| Q5 | Top 10 highest-value customers? | Composite scoring: Platinum multi-order |
| Q6 | Satisfaction by segment x channel? | Platinum Web: 4.31 avg |
| Q7 | Value tier distribution? | 38% Loyal, 24% Champion |
| Q8 | Revenue by country? | USA: 45% of revenue |
| Q9 | Peak sign-up weekday? | Tuesday leads |
| Q10 | Cohort spend by join year? | 2022 cohort: highest avg spend |

</details>

```python
result = (
    df.groupby(["segment", "channel"])
      .agg(avg_spend=("annual_spend", "mean"),
           total_revenue=("revenue", "sum"),
           customers=("customer_id", "count"))
      .reset_index()
      .pivot_table("avg_spend", "segment", "channel")
)
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| **Language** | Python | 3.10+ | Core |
| **Data** | Pandas | 2.0+ | Manipulation & analysis |
| **Numeric** | NumPy | 1.24+ | Array operations |
| **ML** | Scikit-Learn | 1.3+ | Regression · scaling · CV |
| **Stats** | SciPy | 1.10+ | Hypothesis tests |
| **Viz (static)** | Matplotlib | 3.7+ | Custom charts |
| **Viz (stats)** | Seaborn | 0.12+ | Statistical plots |
| **App** | Streamlit | 1.32+ | Interactive deployment |
| **Notebooks** | Jupyter | 7.0+ | Task notebooks |

</div>

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/data-science-portfolio.git
cd data-science-portfolio

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit app (all 10 tasks)
streamlit run app.py

# Or open individual notebooks
jupyter lab notebooks/
```

---

## 📁 Project Structure

```
data-science-portfolio/
|
+-- notebooks/
|   +-- task01_statistical_analysis.ipynb
|   +-- task02_data_cleaning.ipynb
|   +-- task03_exploratory_eda.ipynb
|   +-- task04_linear_regression.ipynb
|   +-- task05_seaborn_visualisation.ipynb
|   +-- task06_hypothesis_testing.ipynb
|   +-- task07_matplotlib_visualisation.ipynb
|   +-- task08_correlation_matrix.ipynb
|   +-- task09_pandas_data_cleaning.ipynb
|   +-- task10_data_manipulation.ipynb
|
+-- app.py               <- Streamlit app (all 10 tasks)
+-- requirements.txt
+-- README.md
```

---

## 📊 Skills Coverage

<div align="center">

```
Data Wrangling        ████████████████████ 100%
Statistical Analysis  ██████████████████░░  90%
Data Visualisation    ████████████████████ 100%
Machine Learning      █████████████░░░░░░░  65%
Hypothesis Testing    ███████████████████░  95%
Feature Engineering   ████████████████░░░░  80%
Data Cleaning         ████████████████████ 100%
Interactive Dashboards███████████████████░  95%
```

</div>

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Star this repository if you found it useful!**

[![GitHub stars](https://img.shields.io/github/stars/your-username/data-science-portfolio?style=social)](https://github.com/your-username/data-science-portfolio)
[![GitHub forks](https://img.shields.io/github/forks/your-username/data-science-portfolio?style=social)](https://github.com/your-username/data-science-portfolio)

*Built with Python · Pandas · Streamlit · heart*

</div>