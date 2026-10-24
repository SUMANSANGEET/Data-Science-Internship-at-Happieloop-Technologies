"""
Data Science Tasks 1–10  |  Streamlit Dashboard
================================================
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import (pearsonr, spearmanr, ttest_ind, ttest_1samp,
                          f_oneway, chi2_contingency, mannwhitneyu)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Science Lab | Tasks 01–10",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour palette ───────────────────────────────────────────────────────────
C = dict(blue='#2563EB', teal='#0891B2', green='#059669',
         amber='#D97706', red='#DC2626', violet='#7C3AED',
         pink='#DB2777',  gray='#6B7280', dark='#1E293B',
         lime='#65A30D',  orange='#EA580C')

# ── Professional branding & responsive dashboard styling ────────────────────
st.markdown("""
<style>
:root { --ds-ink:#10233f; --ds-blue:#2563eb; --ds-cyan:#06b6d4; --ds-muted:#64748b; }
.block-container { padding-top: 1.35rem; padding-bottom: 2.5rem; max-width: 1500px; }
[data-testid="stSidebar"] { border-right: 1px solid rgba(100,116,139,.20); }
.ds-brand { display:flex; align-items:center; gap:11px; padding:10px 4px 16px; }
.ds-mark { width:48px; height:48px; border-radius:14px; display:flex; align-items:center;
           justify-content:center; color:white; font-size:17px; font-weight:850; letter-spacing:-1px;
           background:linear-gradient(135deg,#2563eb 0%,#06b6d4 100%);
           box-shadow:0 7px 18px rgba(37,99,235,.24); }
.ds-brand-name { color:var(--ds-ink); font-size:17px; font-weight:800; line-height:1.1; }
.ds-brand-sub { color:var(--ds-muted); font-size:10px; letter-spacing:1.5px; font-weight:700; margin-top:5px; }
.ds-hero { padding:25px 28px; border-radius:20px; color:white; margin:4px 0 22px;
           background:linear-gradient(115deg,#10233f 0%,#173f78 58%,#087e9b 100%);
           border:1px solid rgba(255,255,255,.12); }
.ds-eyebrow { text-transform:uppercase; letter-spacing:2px; font-size:11px; font-weight:800;
              color:#9be7f4; margin-bottom:8px; }
.ds-hero h1 { color:white !important; font-size:clamp(25px,3vw,38px) !important;
              line-height:1.12 !important; margin:0 0 10px !important; }
.ds-hero p { color:#dceafe; margin:0; font-size:15px; max-width:850px; }
.ds-pill { display:inline-block; margin-top:17px; padding:6px 10px; border-radius:999px;
           color:#e0f2fe; background:rgba(255,255,255,.12); font-size:11px; font-weight:700; }
.ds-section { color:var(--ds-ink); font-size:20px; font-weight:800; margin:8px 0 12px; }
.ds-card { border:1px solid #e2e8f0; border-radius:14px; padding:15px 16px; background:rgba(248,250,252,.8);
           min-height:124px; margin-bottom:12px; }
.ds-card .icon { font-size:21px; }
.ds-card .name { color:#1e293b; font-size:15px; font-weight:800; margin:7px 0 5px; }
.ds-card .desc { color:#64748b; font-size:12px; line-height:1.5; }
div[data-testid="stMetric"] { border:1px solid #e2e8f0; padding:14px 16px; border-radius:13px;
                              background:rgba(248,250,252,.75); }
div[data-testid="stMetricLabel"] { color:#64748b; }
div[data-testid="stMetricValue"] { color:#10233f; }
.stTabs [data-baseweb="tab-list"] { gap:8px; }
.stTabs [data-baseweb="tab"] { border-radius:9px 9px 0 0; }
@media (max-width: 700px) {
 .ds-hero { padding:20px; border-radius:15px; }
 .ds-card { min-height:unset; }
}
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({'figure.facecolor': 'white',
                     'axes.spines.top': False, 'axes.spines.right': False})

# ════════════════════════════════════════════════════════════════════════════
# DATA GENERATORS  (cached so they don't rerun on every widget interaction)
# ════════════════════════════════════════════════════════════════════════════
@st.cache_data
def make_iris():
    np.random.seed(0)
    def make_class(n, sl_mu, sw_mu, pl_mu, pw_mu):
        return pd.DataFrame({
            'sepal_length': np.random.normal(sl_mu, 0.35, n),
            'sepal_width' : np.random.normal(sw_mu, 0.38, n),
            'petal_length': np.random.normal(pl_mu, 0.46, n),
            'petal_width' : np.random.normal(pw_mu, 0.20, n),
        })
    s  = make_class(50, 5.01, 3.42, 1.46, 0.24); s['species']='Setosa'
    v  = make_class(50, 5.94, 2.77, 4.26, 1.33); v['species']='Versicolor'
    vi = make_class(50, 6.59, 2.97, 5.55, 2.03); vi['species']='Virginica'
    return pd.concat([s, v, vi], ignore_index=True)

@st.cache_data
def make_retail(n=1_200, seed=42):
    np.random.seed(seed)
    SEGS  = ['Bronze','Silver','Gold','Platinum']
    REGS  = ['North','South','East','West']
    CATS  = ['Electronics','Clothing','Home','Sports','Books']
    CHANS = ['Web','Mobile','In-Store','Phone']
    GENS  = ['Male','Female','Other']
    seg_inc = {'Bronze':31000,'Silver':53000,'Gold':79000,'Platinum':125000}
    seg_ord = {'Bronze':4,'Silver':9,'Gold':16,'Platinum':28}
    segs = np.random.choice(SEGS, n, p=[.35,.30,.22,.13])
    rows = []
    for i, seg in enumerate(segs):
        inc = max(np.random.normal(seg_inc[seg], seg_inc[seg]*.18), 9000)
        spd = max(800+.21*inc+np.random.normal(0,1800), 80)
        rows.append({
            'customer_id' : 1001+i,
            'gender'      : np.random.choice(GENS, p=[.47,.48,.05]),
            'age'         : int(np.clip(np.random.normal(38,12),18,74)),
            'segment'     : seg,
            'region'      : np.random.choice(REGS),
            'category'    : np.random.choice(CATS, p=[.28,.24,.18,.17,.13]),
            'channel'     : np.random.choice(CHANS, p=[.40,.35,.18,.07]),
            'annual_income': round(inc,2),
            'annual_spend' : round(spd,2),
            'num_orders'   : int(np.random.poisson(seg_ord[seg])),
            'satisfaction' : int(np.random.choice([1,2,3,4,5],p=[.04,.09,.20,.42,.25])),
            'tenure_years' : round(np.clip(np.random.exponential(3.5),.1,15),1),
            'discount_pct' : float(np.random.choice([0,5,10,15,20,25],
                                   p=[.40,.20,.18,.12,.06,.04])),
            'unit_price'   : round(np.random.uniform(10,800),2),
            'quantity'     : int(np.random.choice([1,2,3,4,5],p=[.50,.25,.12,.08,.05])),
            'product_rating': round(np.random.uniform(2.5,5.0),1),
            'join_date'    : pd.Timestamp('2019-01-01')+pd.Timedelta(
                                days=int(np.random.uniform(0,1826))),
            'return_rate'  : round(np.random.beta(2,8),4),
        })
    df = pd.DataFrame(rows)
    df['revenue']         = (df['unit_price']*df['quantity']*(1-df['discount_pct']/100)).round(2)
    df['avg_order_value'] = (df['annual_spend']/(df['num_orders']+1)).round(2)
    df['discount_used']   = (df['discount_pct']>0).astype(int)
    df['join_year']       = df['join_date'].dt.year
    df['join_month']      = df['join_date'].dt.month
    df['loyalty_score']   = (df['tenure_years']*df['num_orders']/10).round(2)
    df['spend_rate']      = (df['annual_spend']/df['annual_income']*100).round(3)
    df['segment'] = pd.Categorical(df['segment'],
                                   categories=['Bronze','Silver','Gold','Platinum'],
                                   ordered=True)
    return df

# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ════════════════════════════════════════════════════════════════════════════
TASKS = {
    "🏠 Home":                          "home",
    "📊 Task 1 – Statistical Analysis": "t1",
    "🧹 Task 2 – Data Cleaning":        "t2",
    "🔭 Task 3 – Exploratory EDA":      "t3",
    "📈 Task 4 – Linear Regression":    "t4",
    "🎨 Task 5 – Seaborn Viz":          "t5",
    "🧪 Task 6 – Hypothesis Testing":   "t6",
    "🖼️ Task 7 – Matplotlib Viz":       "t7",
    "🔗 Task 8 – Correlation Matrix":   "t8",
    "🧼 Task 9 – Pandas Data Cleaning": "t9",
    "🐼 Task 10 – Data Manipulation":   "t10",
}

with st.sidebar:
    # Original, self-contained text logo: no external image or asset required.
    st.markdown("""
    <div class="ds-brand">
      <div class="ds-mark">DS<span style="font-size:10px">+</span></div>
      <div>
        <div class="ds-brand-name">Data Science Lab</div>
        <div class="ds-brand-sub">ANALYTICS · INSIGHTS</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Explore the workspace**")
    selection = st.radio("Navigate", list(TASKS.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Built with Python · Streamlit")
    st.caption("10 interactive learning modules")

page = TASKS[selection]

# ════════════════════════════════════════════════════════════════════════════
# HOME
# ════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown("""
    <section class="ds-hero">
      <div class="ds-eyebrow">Interactive analytics workspace</div>
      <h1>Data Science Tasks 01–10</h1>
      <p>Explore data quality, statistical evidence, visual patterns and predictive modelling in one recruiter-friendly portfolio dashboard.</p>
      <span class="ds-pill">PYTHON&nbsp; / &nbsp;STATISTICS&nbsp; / &nbsp;MACHINE LEARNING</span>
    </section>
    """, unsafe_allow_html=True)

    # These are generated demonstration datasets, not live company/business data.
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Learning modules", "10", help="Ten task areas accessible from the sidebar.")
    k2.metric("Core toolset", "Python + SQL", help="This app primarily demonstrates Python analytics workflows.")
    k3.metric("Analysis modes", "20+", help="Multiple statistical tests, chart types and data operations.")
    k4.metric("Data status", "Demo data", help="The built-in Iris and retail datasets are generated for demonstration.")

    st.markdown('<div class="ds-section">Explore the modules</div>', unsafe_allow_html=True)
    st.caption("Choose a module below or use the sidebar navigation. Interactive controls let you change variables, methods and model settings.")
    cols = st.columns(3)
    tasks_info = [
        ("📊","Task 1","Statistical Analysis","Iris dataset · ANOVA · normality tests · 6 chart types"),
        ("🧹","Task 2","Data Cleaning Pipeline","12 quality issues · imputation · encoding · validation"),
        ("🔭","Task 3","Exploratory EDA","Univariate · bivariate · heatmaps · temporal trends"),
        ("📈","Task 4","Linear Regression","Simple → Ridge/Lasso · R² · residual diagnostics"),
        ("🎨","Task 5","Seaborn Visualisation","20+ chart types · facets · custom themes"),
        ("🧪","Task 6","Hypothesis Testing","8 statistical tests · effect sizes · power analysis"),
        ("🖼️","Task 7","Matplotlib Visualisation","30+ chart types · GridSpec · custom styles"),
        ("🔗","Task 8","Correlation Matrix","Pearson/Spearman/Kendall · clustering · partial corr"),
        ("🧼","Task 9","Pandas Data Cleaning","Type fixes · dedup · outliers · 20 validation checks"),
        ("🐼","Task 10","Data Manipulation","Filter · sort · groupby · merge · reshape · windows"),
    ]
    for i, (icon, title, subtitle, desc) in enumerate(tasks_info):
        with cols[i % 3]:
            st.markdown(f"""
<div class="ds-card">
<div class="icon">{icon}</div>
<div class="name">{title} · {subtitle}</div>
<div class="desc">{desc}</div>
</div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 1 — STATISTICAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "t1":
    st.title("📊 Task 1 — Statistical Analysis")
    st.markdown("**Dataset:** Iris (150 samples × 4 features × 3 species)")
    iris = make_iris()

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Dataset","📐 Descriptive Stats","📊 Visualisations","🧪 Tests"])

    with tab1:
        st.subheader("Iris Dataset")
        st.dataframe(iris.round(3), use_container_width=True, height=300)
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", len(iris)); c2.metric("Features", 4); c3.metric("Species", 3)

    with tab2:
        feat = st.selectbox("Feature", ['sepal_length','sepal_width','petal_length','petal_width'])
        st.subheader(f"Statistics for `{feat}`")
        stats_df = iris.groupby('species')[feat].agg(
            ['mean','median','std','min','max',
             lambda x: x.quantile(.25), lambda x: x.quantile(.75)]
        ).round(4)
        stats_df.columns = ['Mean','Median','Std','Min','Max','Q1','Q3']
        st.dataframe(stats_df, use_container_width=True)

    with tab3:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("Iris — Statistical Visualisations", fontsize=13, fontweight='bold')
        features = ['sepal_length','sepal_width','petal_length','petal_width']
        palette  = {'Setosa':C['blue'],'Versicolor':C['green'],'Virginica':C['red']}

        # Histogram
        for sp, col in palette.items():
            axes[0,0].hist(iris[iris['species']==sp]['petal_length'],
                           bins=15, alpha=0.65, color=col, label=sp, edgecolor='white')
        axes[0,0].set_title('Petal Length Distribution'); axes[0,0].legend(fontsize=8)

        # Box plots
        sns.boxplot(data=iris, x='species', y='petal_width', palette=list(palette.values()),
                    ax=axes[0,1]); axes[0,1].set_title('Petal Width Box Plot')

        # Violin
        sns.violinplot(data=iris, x='species', y='sepal_length',
                       palette=list(palette.values()), ax=axes[0,2])
        axes[0,2].set_title('Sepal Length Violin')

        # Scatter
        for sp, col in palette.items():
            sub = iris[iris['species']==sp]
            axes[1,0].scatter(sub['sepal_length'], sub['petal_length'],
                              alpha=0.65, s=30, color=col, label=sp)
        axes[1,0].set_xlabel('Sepal Length'); axes[1,0].set_ylabel('Petal Length')
        axes[1,0].set_title('Sepal vs Petal Length'); axes[1,0].legend(fontsize=8)

        # Correlation heatmap
        corr = iris[features].corr()
        sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-1, vmax=1, ax=axes[1,1], cbar_kws={'shrink':.75})
        axes[1,1].set_title('Feature Correlation Matrix')

        # Pair mean bar
        means = iris.groupby('species')[features].mean()
        means.T.plot(kind='bar', ax=axes[1,2], color=list(palette.values()),
                     alpha=0.82, edgecolor='white', width=0.7)
        axes[1,2].set_title('Feature Means by Species')
        axes[1,2].tick_params(axis='x', rotation=15)
        axes[1,2].legend(fontsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab4:
        st.subheader("One-Way ANOVA — All Features")
        results = []
        for feat in ['sepal_length','sepal_width','petal_length','petal_width']:
            groups = [iris[iris['species']==sp][feat].values for sp in iris['species'].unique()]
            F, p   = f_oneway(*groups)
            results.append({'Feature':feat, 'F-statistic':round(F,4),
                             'p-value':round(p,6),
                             'Significant (p<0.05)': '✅ Yes' if p<0.05 else '❌ No'})
        st.dataframe(pd.DataFrame(results), use_container_width=True)

        st.subheader("Normality Tests (Shapiro-Wilk)")
        norm_res = []
        for sp in iris['species'].unique():
            for feat in ['petal_length','petal_width']:
                sub = iris[iris['species']==sp][feat]
                W, p = stats.shapiro(sub)
                norm_res.append({'Species':sp,'Feature':feat,
                                  'W':round(W,4),'p-value':round(p,4),
                                  'Normal?': '✅' if p>0.05 else '❌'})
        st.dataframe(pd.DataFrame(norm_res), use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 2 — DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
elif page == "t2":
    st.title("🧹 Task 2 — Data Cleaning Pipeline")
    df_raw = make_retail(seed=99)

    # Inject dirt
    @st.cache_data
    def inject_dirt(df):
        dirty = df.copy()
        n = len(dirty)
        for col, pct in [('annual_income',.08),('annual_spend',.07),
                          ('satisfaction',.10),('tenure_years',.06)]:
            idx = np.random.RandomState(77).choice(n, int(n*pct), replace=False)
            dirty.loc[idx, col] = np.nan
        dup_idx = np.random.RandomState(77).choice(n, 25, replace=False)
        dirty = pd.concat([dirty, dirty.iloc[dup_idx]], ignore_index=True)
        bad_age = np.random.RandomState(77).choice(len(dirty), 30, replace=False)
        dirty.loc[bad_age[:15], 'age'] = np.random.choice([-5,150,999],15)
        dirty.loc[bad_age[15:], 'satisfaction'] = np.random.choice([0,6,7],15)
        return dirty

    dirty = inject_dirt(df_raw)

    tab1, tab2, tab3 = st.tabs(["🔴 Raw Data Issues", "🔧 After Cleaning", "📊 Before vs After"])

    with tab1:
        st.subheader("Raw Data — Issue Checklist")
        issues = {
            'Total rows': len(dirty),
            'Missing cells': dirty.isnull().sum().sum(),
            'Duplicate rows': int(dirty.duplicated().sum()),
            'Invalid Age (<18 or >100)': int(((pd.to_numeric(dirty['age'],errors='coerce')<18)|
                                               (pd.to_numeric(dirty['age'],errors='coerce')>100)).sum()),
            'Invalid Satisfaction (not 1-5)': int(((pd.to_numeric(dirty['satisfaction'],errors='coerce')<1)|
                                                    (pd.to_numeric(dirty['satisfaction'],errors='coerce')>5)).sum()),
        }
        c1,c2,c3,c4,c5 = st.columns(5)
        for (k,v), col in zip(issues.items(),[c1,c2,c3,c4,c5]):
            col.metric(k, f"{v:,}")
        st.dataframe(dirty.head(20), use_container_width=True, height=300)

        st.subheader("Missing Values")
        miss = dirty.isnull().mean()*100
        miss = miss[miss>0].sort_values(ascending=False)
        fig2, ax2 = plt.subplots(figsize=(9,3))
        ax2.barh(miss.index, miss.values, color=C['red'], alpha=0.8, edgecolor='white')
        ax2.set_xlabel('Missing %')
        ax2.set_title('Missing Rate per Column')
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab2:
        @st.cache_data
        def clean_data(dirty):
            df = dirty.copy()
            df.drop_duplicates(inplace=True)
            df.reset_index(drop=True, inplace=True)
            df['annual_income'] = pd.to_numeric(df['annual_income'],errors='coerce')
            df['annual_spend']  = pd.to_numeric(df['annual_spend'], errors='coerce')
            df['annual_income'] = df.groupby('segment')['annual_income'].transform(
                lambda x: x.fillna(x.median()))
            df['annual_spend']  = df.groupby('segment')['annual_spend'].transform(
                lambda x: x.fillna(x.median()))
            df['satisfaction']  = pd.to_numeric(df['satisfaction'],errors='coerce')
            df['tenure_years']  = pd.to_numeric(df['tenure_years'],errors='coerce')
            df['satisfaction']  = df['satisfaction'].fillna(df['satisfaction'].mode()[0])
            df['tenure_years']  = df['tenure_years'].fillna(df['tenure_years'].median())
            df.loc[(df['age']<18)|(df['age']>100),'age'] = np.nan
            df['age'] = df['age'].fillna(df['age'].median()).round().astype(int)
            df.loc[(df['satisfaction']<1)|(df['satisfaction']>5),'satisfaction'] = df['satisfaction'].mode()[0]
            for col in ['annual_income','annual_spend']:
                Q1,Q3 = df[col].quantile(.25), df[col].quantile(.75)
                df[col] = df[col].clip(Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1))
            return df

        df_clean = clean_data(dirty)
        st.success(f"✅ Cleaning complete — {len(dirty)-len(df_clean):,} rows removed")
        st.subheader("Clean Dataset")
        st.dataframe(df_clean.head(20), use_container_width=True, height=300)
        c1,c2,c3 = st.columns(3)
        c1.metric("Clean Rows", f"{len(df_clean):,}")
        c2.metric("Missing Cells", f"{df_clean.isnull().sum().sum()}")
        c3.metric("Duplicates", f"{df_clean.duplicated().sum()}")

    with tab3:
        st.subheader("Distribution Comparison — Annual Income")
        fig3, axes3 = plt.subplots(1,2,figsize=(12,4))
        df_clean2 = clean_data(dirty)
        axes3[0].hist(pd.to_numeric(dirty['annual_income'],errors='coerce').dropna(),
                      bins=40, color=C['red'], alpha=0.75, edgecolor='white', density=True)
        axes3[0].set_title('BEFORE — Annual Income', color=C['red'], fontweight='bold')
        axes3[1].hist(df_clean2['annual_income'].dropna(), bins=40,
                      color=C['green'], alpha=0.75, edgecolor='white', density=True)
        axes3[1].set_title('AFTER — Annual Income', color=C['green'], fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 3 — EDA
# ════════════════════════════════════════════════════════════════════════════
elif page == "t3":
    st.title("🔭 Task 3 — Exploratory Data Analysis")
    df = make_retail()

    tab1,tab2,tab3,tab4 = st.tabs(["📋 Overview","📊 Univariate","🔀 Bivariate","🌡️ Correlation"])

    with tab1:
        st.dataframe(df[['customer_id','segment','region','category','channel',
                          'annual_income','annual_spend','num_orders','satisfaction',
                          'tenure_years']].head(15), use_container_width=True)
        st.subheader("Summary Statistics")
        st.dataframe(df[['annual_income','annual_spend','num_orders',
                          'satisfaction','tenure_years']].describe().round(2),
                     use_container_width=True)

    with tab2:
        col_choice = st.selectbox("Select column",
                                  ['annual_income','annual_spend','num_orders',
                                   'satisfaction','tenure_years','age'])
        fig_u, axes_u = plt.subplots(1,3, figsize=(14,4))
        data = df[col_choice].dropna()
        axes_u[0].hist(data, bins=35, color=C['blue'], alpha=0.80, edgecolor='white')
        axes_u[0].set_title(f'{col_choice} — Histogram')
        axes_u[1].boxplot(data, patch_artist=True,
                          boxprops=dict(facecolor=C['blue'],alpha=0.7),
                          medianprops=dict(color='white',lw=2))
        axes_u[1].set_title('Box Plot')
        stats.probplot(data, plot=axes_u[2])
        axes_u[2].set_title('Q-Q Plot')
        plt.tight_layout()
        st.pyplot(fig_u, use_container_width=True)

        st.subheader("Categorical Distributions")
        fig_cat, axes_cat = plt.subplots(1,3,figsize=(14,4))
        for ax, col in zip(axes_cat, ['segment','region','channel']):
            counts = df[col].value_counts()
            ax.bar(counts.index, counts.values,
                   color=[C['blue'],C['teal'],C['green'],C['amber']][:len(counts)],
                   alpha=0.85, edgecolor='white')
            ax.set_title(f'{col} Distribution')
            ax.tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig_cat, use_container_width=True)

    with tab3:
        x_col = st.selectbox("X axis", ['annual_income','num_orders','tenure_years'], key='bx')
        y_col = st.selectbox("Y axis", ['annual_spend','satisfaction','revenue'], key='by')
        hue   = st.selectbox("Color by", ['segment','region','channel'], key='bh')

        fig_b, ax_b = plt.subplots(figsize=(10,5))
        palette_map = {'Bronze':C['gray'],'Silver':C['blue'],'Gold':C['amber'],'Platinum':C['violet'],
                       'North':C['blue'],'South':C['green'],'East':C['teal'],'West':C['amber'],
                       'Web':C['blue'],'Mobile':C['teal'],'In-Store':C['green'],'Phone':C['amber']}
        for cat in df[hue].unique():
            sub = df[df[hue]==cat].sample(min(150,(df[hue]==cat).sum()), random_state=1)
            ax_b.scatter(sub[x_col], sub[y_col], alpha=0.45, s=22,
                         color=palette_map.get(str(cat), C['gray']), label=str(cat))
        m, b_v = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
        xf = np.linspace(df[x_col].min(), df[x_col].max(), 200)
        ax_b.plot(xf, m*xf+b_v, color=C['dark'], lw=2.5, ls='--')
        r = df[x_col].corr(df[y_col])
        ax_b.set_title(f'{x_col} vs {y_col}  (r={r:.3f})', fontweight='bold')
        ax_b.set_xlabel(x_col); ax_b.set_ylabel(y_col)
        ax_b.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)

    with tab4:
        num_cols = ['annual_income','annual_spend','num_orders','satisfaction',
                    'tenure_years','loyalty_score','spend_rate']
        corr = df[num_cols].corr()
        fig_c, ax_c = plt.subplots(figsize=(9,7))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-1, vmax=1, ax=ax_c, cbar_kws={'shrink':.75},
                    linewidths=0.5)
        ax_c.set_title('Correlation Matrix', fontweight='bold', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 4 — LINEAR REGRESSION
# ════════════════════════════════════════════════════════════════════════════
elif page == "t4":
    st.title("📈 Task 4 — Linear Regression")
    df = make_retail()

    tab1,tab2,tab3 = st.tabs(["🔢 Simple Regression","📦 Multiple Regression","🔍 Diagnostics"])

    with tab1:
        x_feat = st.selectbox("Predictor (X)", ['annual_income','num_orders','tenure_years'])
        y_feat = 'annual_spend'

        X = df[[x_feat]].values
        y = df[y_feat].values
        X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.2,random_state=42)
        mdl = LinearRegression().fit(X_tr,y_tr)
        y_pred = mdl.predict(X_te)
        r2  = r2_score(y_te,y_pred)
        rmse= np.sqrt(mean_squared_error(y_te,y_pred))

        c1,c2,c3 = st.columns(3)
        c1.metric("R² (test)", f"{r2:.4f}")
        c2.metric("RMSE", f"${rmse:,.0f}")
        c3.metric("Slope", f"{mdl.coef_[0]:.4f}")

        fig_r, axes_r = plt.subplots(1,2,figsize=(12,5))
        axes_r[0].scatter(X_te, y_te, alpha=0.35, s=18, color=C['blue'])
        xf = np.linspace(X_te.min(),X_te.max(),200).reshape(-1,1)
        axes_r[0].plot(xf, mdl.predict(xf), color=C['red'], lw=2.5, label=f'R²={r2:.3f}')
        axes_r[0].set_xlabel(x_feat); axes_r[0].set_ylabel(y_feat)
        axes_r[0].set_title('Actual vs Regression Line'); axes_r[0].legend()
        resid = y_te - y_pred
        axes_r[1].scatter(y_pred, resid, alpha=0.35, s=18, color=C['teal'])
        axes_r[1].axhline(0, color=C['red'], lw=2, ls='--')
        axes_r[1].set_xlabel('Predicted'); axes_r[1].set_ylabel('Residuals')
        axes_r[1].set_title('Residual Plot')
        plt.tight_layout()
        st.pyplot(fig_r, use_container_width=True)

    with tab2:
        feats = st.multiselect("Predictor features",
                               ['annual_income','num_orders','tenure_years',
                                'age','loyalty_score','discount_pct'],
                               default=['annual_income','num_orders','tenure_years'])
        if len(feats) >= 2:
            alpha_val = st.slider("Regularisation α", 0.0, 10.0, 1.0, 0.1)
            model_type = st.radio("Model", ["OLS","Ridge","Lasso"], horizontal=True)

            X_m = df[feats].fillna(df[feats].median())
            y_m = df['annual_spend']
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_m)
            X_tr,X_te,y_tr,y_te = train_test_split(X_s,y_m,test_size=0.2,random_state=42)
            mdl_m = (Ridge(alpha=alpha_val) if model_type=="Ridge"
                     else Lasso(alpha=alpha_val) if model_type=="Lasso"
                     else LinearRegression())
            mdl_m.fit(X_tr,y_tr)
            r2_m  = r2_score(y_te, mdl_m.predict(X_te))
            rmse_m= np.sqrt(mean_squared_error(y_te, mdl_m.predict(X_te)))
            cv_r2 = cross_val_score(mdl_m, X_s, y_m, cv=5, scoring='r2').mean()

            c1,c2,c3 = st.columns(3)
            c1.metric("Test R²", f"{r2_m:.4f}")
            c2.metric("RMSE", f"${rmse_m:,.0f}")
            c3.metric("5-Fold CV R²", f"{cv_r2:.4f}")

            coef_df = pd.DataFrame({'Feature':feats,'Coefficient':mdl_m.coef_}).sort_values(
                'Coefficient', key=abs, ascending=True)
            fig_c, ax_c = plt.subplots(figsize=(8,4))
            colors = [C['green'] if v>0 else C['red'] for v in coef_df['Coefficient']]
            ax_c.barh(coef_df['Feature'], coef_df['Coefficient'],
                      color=colors, alpha=0.85, edgecolor='white')
            ax_c.axvline(0, color='black', lw=1)
            ax_c.set_title(f'{model_type} Regression Coefficients', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_c, use_container_width=True)

    with tab3:
        X_d = df[['annual_income']].values
        y_d = df['annual_spend'].values
        mdl_d = LinearRegression().fit(X_d, y_d)
        resid_d = y_d - mdl_d.predict(X_d)

        fig_d, axes_d = plt.subplots(2,2,figsize=(12,8))
        fig_d.suptitle('Regression Diagnostics', fontweight='bold')
        axes_d[0,0].scatter(mdl_d.predict(X_d), resid_d, alpha=0.3, s=15, color=C['blue'])
        axes_d[0,0].axhline(0, color=C['red'], lw=2, ls='--')
        axes_d[0,0].set_title('Residuals vs Fitted'); axes_d[0,0].set_xlabel('Fitted')
        stats.probplot(resid_d, plot=axes_d[0,1])
        axes_d[0,1].set_title('Normal Q-Q Plot')
        axes_d[1,0].hist(resid_d, bins=40, color=C['teal'], alpha=0.78, edgecolor='white')
        axes_d[1,0].set_title('Residual Distribution')
        y_pred_d = mdl_d.predict(X_d)
        axes_d[1,1].scatter(y_d, y_pred_d, alpha=0.3, s=15, color=C['violet'])
        lims = [min(y_d.min(),y_pred_d.min()), max(y_d.max(),y_pred_d.max())]
        axes_d[1,1].plot(lims, lims, color=C['red'], lw=2, ls='--')
        axes_d[1,1].set_title('Actual vs Predicted')
        axes_d[1,1].set_xlabel('Actual'); axes_d[1,1].set_ylabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig_d, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 5 — SEABORN VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "t5":
    st.title("🎨 Task 5 — Seaborn Visualisation")
    df = make_retail()

    viz_type = st.selectbox("Choose visualisation",
        ["Distribution Grid","Violin + Strip","Regression Plots",
         "Categorical Plots","Heatmap Gallery","Pair Plot (sample)"])

    if viz_type == "Distribution Grid":
        fig, axes = plt.subplots(2,3,figsize=(14,8))
        fig.suptitle('Distribution Analysis', fontweight='bold')
        for ax, col in zip(axes.flat[:5], ['annual_income','annual_spend','num_orders',
                                            'tenure_years','product_rating']):
            sns.histplot(data=df, x=col, hue='segment', ax=ax, kde=True,
                         palette=[C['gray'],C['blue'],C['amber'],C['violet']])
            ax.set_title(col)
        sns.kdeplot(data=df, x='annual_income', hue='segment', ax=axes[1,2],
                    palette=[C['gray'],C['blue'],C['amber'],C['violet']])
        axes[1,2].set_title('Income KDE by Segment')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif viz_type == "Violin + Strip":
        feat = st.selectbox("Feature", ['annual_spend','annual_income','num_orders'])
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        sns.violinplot(data=df, x='segment', y=feat, order=['Bronze','Silver','Gold','Platinum'],
                       palette=[C['gray'],C['blue'],C['amber'],C['violet']], ax=axes[0])
        axes[0].set_title(f'Violin — {feat}')
        sns.stripplot(data=df.sample(300,random_state=1), x='segment', y=feat,
                      order=['Bronze','Silver','Gold','Platinum'],
                      palette=[C['gray'],C['blue'],C['amber'],C['violet']],
                      alpha=0.4, jitter=True, ax=axes[1])
        axes[1].set_title(f'Strip — {feat}')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif viz_type == "Regression Plots":
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        sns.regplot(data=df.sample(400,random_state=1), x='annual_income', y='annual_spend',
                    scatter_kws={'alpha':0.3,'s':18}, line_kws={'color':C['red'],'lw':2.5},
                    ax=axes[0])
        axes[0].set_title('Income → Spend (Linear)')
        sns.residplot(data=df.sample(400,random_state=1), x='annual_income', y='annual_spend',
                      scatter_kws={'alpha':0.3,'s':18}, ax=axes[1])
        axes[1].set_title('Residual Plot')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif viz_type == "Categorical Plots":
        fig, axes = plt.subplots(1,3,figsize=(15,5))
        sns.barplot(data=df, x='segment', y='satisfaction',
                    order=['Bronze','Silver','Gold','Platinum'],
                    palette=[C['gray'],C['blue'],C['amber'],C['violet']],
                    errorbar='se', ax=axes[0]); axes[0].set_title('Avg Satisfaction by Segment')
        sns.countplot(data=df, x='channel', hue='segment',
                      palette=[C['gray'],C['blue'],C['amber'],C['violet']], ax=axes[1])
        axes[1].set_title('Channel by Segment'); axes[1].legend(fontsize=8)
        sns.pointplot(data=df, x='segment', y='num_orders',
                      order=['Bronze','Silver','Gold','Platinum'],
                      palette=[C['gray'],C['blue'],C['amber'],C['violet']],
                      ax=axes[2]); axes[2].set_title('Avg Orders by Segment')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif viz_type == "Heatmap Gallery":
        fig, axes = plt.subplots(1,2,figsize=(14,6))
        pivot = df.pivot_table('annual_spend','segment','region',aggfunc='mean').round(0)
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0],
                    linewidths=0.5, cbar_kws={'shrink':.75})
        axes[0].set_title('Avg Spend — Segment × Region')
        corr_m = df[['annual_income','annual_spend','num_orders',
                      'satisfaction','tenure_years']].corr()
        mask = np.triu(np.ones_like(corr_m, dtype=bool))
        sns.heatmap(corr_m, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-1, vmax=1, ax=axes[1], linewidths=0.5)
        axes[1].set_title('Correlation Heatmap')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif viz_type == "Pair Plot (sample)":
        sub = df[['annual_income','annual_spend','num_orders','satisfaction','segment']].sample(300,random_state=1)
        fig_pp = sns.pairplot(sub, hue='segment', corner=True,
                              palette={'Bronze':C['gray'],'Silver':C['blue'],
                                       'Gold':C['amber'],'Platinum':C['violet']},
                              plot_kws={'alpha':0.4,'s':15},
                              diag_kind='kde')
        fig_pp.fig.suptitle('Pair Plot — Key Features by Segment', y=1.01, fontweight='bold')
        st.pyplot(fig_pp.fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 6 — HYPOTHESIS TESTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "t6":
    st.title("🧪 Task 6 — Hypothesis Testing")
    df = make_retail()

    tab1,tab2,tab3 = st.tabs(["🔬 Run Tests","📊 Test Results","📐 Power Analysis"])

    with tab1:
        test_choice = st.selectbox("Select test", [
            "One-Sample t-test", "Two-Sample t-test (Welch)",
            "One-Way ANOVA", "Chi-Square Test",
            "Mann-Whitney U", "Kruskal-Wallis"])

        if test_choice == "One-Sample t-test":
            col   = st.selectbox("Variable", ['annual_spend','annual_income','num_orders'])
            mu0   = st.number_input("Hypothesised mean (μ₀)", value=float(df[col].mean().round()))
            t, p  = ttest_1samp(df[col].dropna(), mu0)
            st.info(f"H₀: μ = {mu0:.0f}  |  t = {t:.4f}  |  p = {p:.4f}  |  {'**Reject H₀**' if p<0.05 else 'Fail to reject H₀'}")

        elif test_choice == "Two-Sample t-test (Welch)":
            g1 = st.selectbox("Group 1", ['Bronze','Silver','Gold','Platinum'], index=0)
            g2 = st.selectbox("Group 2", ['Bronze','Silver','Gold','Platinum'], index=2)
            col= st.selectbox("Outcome", ['annual_spend','annual_income','satisfaction'])
            a  = df[df['segment']==g1][col].dropna()
            b  = df[df['segment']==g2][col].dropna()
            t, p = ttest_ind(a, b, equal_var=False)
            d    = (a.mean()-b.mean())/np.sqrt((a.std()**2+b.std()**2)/2)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric(f"{g1} mean", f"{a.mean():,.1f}")
            c2.metric(f"{g2} mean", f"{b.mean():,.1f}")
            c3.metric("t-stat", f"{t:.4f}")
            c4.metric("p-value", f"{p:.4f}")
            st.info(f"Cohen's d = {d:.4f}  |  {'**Significant**' if p<0.05 else 'Not significant'}")

        elif test_choice == "One-Way ANOVA":
            col   = st.selectbox("Outcome", ['annual_spend','annual_income','satisfaction','num_orders'])
            groups= [df[df['segment']==s][col].dropna().values for s in ['Bronze','Silver','Gold','Platinum']]
            F, p  = f_oneway(*groups)
            st.info(f"F = {F:.4f}  |  p = {p:.6f}  |  {'**Reject H₀**' if p<0.05 else 'Fail to reject'}")
            fig_a, ax_a = plt.subplots(figsize=(8,4))
            sns.boxplot(data=df, x='segment', y=col,
                        order=['Bronze','Silver','Gold','Platinum'],
                        palette=[C['gray'],C['blue'],C['amber'],C['violet']], ax=ax_a)
            ax_a.set_title(f'ANOVA — {col} by Segment (p={p:.4f})')
            plt.tight_layout(); st.pyplot(fig_a, use_container_width=True)

        elif test_choice == "Chi-Square Test":
            ct = pd.crosstab(df['segment'], df['channel'])
            chi2, p, dof, expected = chi2_contingency(ct)
            st.info(f"χ² = {chi2:.4f}  |  df = {dof}  |  p = {p:.4f}  |  {'Significant' if p<0.05 else 'Not significant'}")
            st.dataframe(ct, use_container_width=True)

        elif test_choice == "Mann-Whitney U":
            g1 = st.selectbox("Group 1", ['Bronze','Silver','Gold','Platinum'], index=0)
            g2 = st.selectbox("Group 2", ['Bronze','Silver','Gold','Platinum'], index=3)
            col= st.selectbox("Variable", ['annual_spend','num_orders','satisfaction'])
            a  = df[df['segment']==g1][col].dropna()
            b  = df[df['segment']==g2][col].dropna()
            U, p = mannwhitneyu(a, b, alternative='two-sided')
            r    = 1 - 2*U/(len(a)*len(b))
            st.info(f"U = {U:.0f}  |  p = {p:.6f}  |  r (effect) = {r:.4f}  |  {'Significant' if p<0.05 else 'Not significant'}")

        elif test_choice == "Kruskal-Wallis":
            col   = st.selectbox("Variable", ['annual_spend','satisfaction','num_orders'])
            groups= [df[df['segment']==s][col].dropna().values for s in ['Bronze','Silver','Gold','Platinum']]
            H, p  = stats.kruskal(*groups)
            st.info(f"H = {H:.4f}  |  p = {p:.6f}  |  {'Significant' if p<0.05 else 'Not significant'}")

    with tab2:
        results = []
        groups_all = [df[df['segment']==s]['annual_spend'].dropna().values
                      for s in ['Bronze','Silver','Gold','Platinum']]
        F,p1 = f_oneway(*groups_all)
        results.append({'Test':'One-Way ANOVA','Variable':'annual_spend by segment',
                         'Statistic':round(F,4),'p-value':round(p1,6),
                         'Significant':'✅' if p1<0.05 else '❌'})
        t2,p2 = ttest_ind(df[df['segment']=='Bronze']['annual_spend'].dropna(),
                           df[df['segment']=='Platinum']['annual_spend'].dropna(), equal_var=False)
        results.append({'Test':'Two-Sample t-test','Variable':'Bronze vs Platinum spend',
                         'Statistic':round(t2,4),'p-value':round(p2,8),
                         'Significant':'✅' if p2<0.05 else '❌'})
        ct2 = pd.crosstab(df['segment'], df['channel'])
        chi2_v,p3,_,__ = chi2_contingency(ct2)
        results.append({'Test':'Chi-Square','Variable':'segment × channel',
                         'Statistic':round(chi2_v,4),'p-value':round(p3,4),
                         'Significant':'✅' if p3<0.05 else '❌'})
        t3,p4 = ttest_ind(df[df['discount_used']==1]['annual_spend'].dropna(),
                           df[df['discount_used']==0]['annual_spend'].dropna(), equal_var=False)
        results.append({'Test':'Two-Sample t-test','Variable':'discount vs no-discount spend',
                         'Statistic':round(t3,4),'p-value':round(p4,4),
                         'Significant':'✅' if p4<0.05 else '❌'})
        H,p5 = stats.kruskal(*groups_all)
        results.append({'Test':'Kruskal-Wallis','Variable':'annual_spend by segment',
                         'Statistic':round(H,4),'p-value':round(p5,6),
                         'Significant':'✅' if p5<0.05 else '❌'})
        st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab3:
        alpha_p = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)
        power_p = st.slider("Desired power (1-β)", 0.70, 0.99, 0.80, 0.01)
        effect  = st.slider("Effect size (Cohen's d)", 0.1, 1.5, 0.5, 0.05)
        from scipy.stats import norm
        z_alpha = norm.ppf(1-alpha_p/2); z_beta = norm.ppf(power_p)
        n_req   = int(np.ceil(2*((z_alpha+z_beta)/effect)**2))
        st.metric("Required sample size per group", f"{n_req:,}")
        sizes = np.arange(0.1, 2.0, 0.05)
        ns    = [int(np.ceil(2*((z_alpha+z_beta)/d)**2)) for d in sizes]
        fig_pw, ax_pw = plt.subplots(figsize=(9,4))
        ax_pw.plot(sizes, ns, color=C['blue'], lw=2.5)
        ax_pw.axvline(effect, color=C['red'], lw=2, ls='--', label=f'd={effect}')
        ax_pw.axhline(n_req, color=C['amber'], lw=2, ls=':', label=f'n={n_req}')
        ax_pw.set_xlabel("Effect size (Cohen's d)")
        ax_pw.set_ylabel("Required n per group")
        ax_pw.set_title(f"Sample Size vs Effect Size (α={alpha_p}, power={power_p})")
        ax_pw.legend(); ax_pw.set_ylim(0, min(ns[0], 3000))
        plt.tight_layout(); st.pyplot(fig_pw, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 7 — MATPLOTLIB VISUALISATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "t7":
    st.title("🖼️ Task 7 — Matplotlib Visualisation")
    df = make_retail()

    chart = st.selectbox("Chart type", [
        "Line Chart + Moving Average", "Grouped Bar Chart",
        "Scatter + Bubble", "Stacked Area Chart",
        "Histogram Gallery", "Pie + Donut",
        "Multi-Subplot Dashboard"])

    if chart == "Line Chart + Moving Average":
        monthly = (df.set_index('join_date').resample('ME')['annual_spend']
                     .mean().reset_index())
        monthly['ma3'] = monthly['annual_spend'].rolling(3,min_periods=1).mean()
        monthly['ma6'] = monthly['annual_spend'].rolling(6,min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(12,5))
        ax.fill_between(range(len(monthly)), monthly['annual_spend'],
                        alpha=0.15, color=C['blue'])
        ax.plot(range(len(monthly)), monthly['annual_spend'],
                color=C['blue'], lw=1.8, alpha=0.7, label='Monthly Avg')
        ax.plot(range(len(monthly)), monthly['ma3'],
                color=C['teal'], lw=2.5, label='3M MA')
        ax.plot(range(len(monthly)), monthly['ma6'],
                color=C['orange'], lw=2.5, ls='--', label='6M MA')
        ax.set_title('Monthly Avg Spend + Rolling Averages', fontweight='bold')
        ax.set_ylabel('Avg Annual Spend ($)'); ax.legend()
        ax.set_xticks(range(0,len(monthly),3))
        ax.set_xticklabels([str(d.year)[2:]+'/'+str(d.month).zfill(2)
                            for d in monthly['join_date'].iloc[::3]], rotation=40)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Grouped Bar Chart":
        pivot = df.groupby(['segment','channel'])['annual_spend'].mean().unstack()
        fig, ax = plt.subplots(figsize=(11,5))
        x    = np.arange(len(pivot.index))
        w    = 0.20
        cols = [C['blue'],C['teal'],C['green'],C['amber']]
        for j,(ch,col) in enumerate(zip(pivot.columns, cols)):
            ax.bar(x+j*w, pivot[ch], w, label=ch, color=col, alpha=0.85, edgecolor='white')
        ax.set_xticks(x+w*1.5); ax.set_xticklabels(pivot.index)
        ax.set_title('Avg Annual Spend — Segment × Channel', fontweight='bold')
        ax.set_ylabel('Avg Spend ($)'); ax.legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Scatter + Bubble":
        sample = df.sample(300, random_state=1)
        fig, axes = plt.subplots(1,2,figsize=(13,5))
        seg_cols = {'Bronze':C['gray'],'Silver':C['blue'],'Gold':C['amber'],'Platinum':C['violet']}
        for seg, color in seg_cols.items():
            s = sample[sample['segment']==seg]
            axes[0].scatter(s['annual_income'], s['annual_spend'],
                            alpha=0.5, s=22, color=color, label=seg)
        axes[0].set_title('Income vs Spend (coloured by segment)')
        axes[0].set_xlabel('Annual Income'); axes[0].set_ylabel('Annual Spend')
        axes[0].legend(fontsize=8)
        for seg, color in seg_cols.items():
            s = sample[sample['segment']==seg]
            axes[1].scatter(s['annual_income'], s['annual_spend'],
                            s=s['num_orders']*8+10, alpha=0.35, color=color, label=seg)
        axes[1].set_title('Bubble — size = num_orders')
        axes[1].set_xlabel('Annual Income'); axes[1].legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Stacked Area Chart":
        monthly_seg = (df.groupby(['join_year','join_month','segment'])['revenue']
                         .sum().reset_index())
        monthly_seg['period'] = monthly_seg['join_year'].astype(str)+'-'+monthly_seg['join_month'].astype(str).str.zfill(2)
        pivot_area = monthly_seg.pivot_table('revenue','period','segment',aggfunc='sum').fillna(0)
        pivot_area = pivot_area[['Bronze','Silver','Gold','Platinum']]
        fig, ax = plt.subplots(figsize=(13,5))
        ax.stackplot(range(len(pivot_area)), pivot_area.T.values,
                     labels=pivot_area.columns,
                     colors=[C['gray'],C['blue'],C['amber'],C['violet']],
                     alpha=0.78)
        ax.set_title('Stacked Revenue by Segment over Time', fontweight='bold')
        ax.set_ylabel('Revenue ($)'); ax.legend(loc='upper left', fontsize=9)
        ax.set_xticks(range(0,len(pivot_area),3))
        ax.set_xticklabels(pivot_area.index[::3], rotation=45, fontsize=7)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Histogram Gallery":
        fig, axes = plt.subplots(2,3,figsize=(14,8))
        fig.suptitle('Histogram Gallery', fontweight='bold')
        cols_h = ['annual_income','annual_spend','num_orders',
                  'tenure_years','product_rating','satisfaction']
        colors_h = [C['blue'],C['teal'],C['green'],C['amber'],C['violet'],C['pink']]
        for ax, col, color in zip(axes.flat, cols_h, colors_h):
            ax.hist(df[col], bins=30, color=color, alpha=0.80, edgecolor='white', density=True)
            from scipy.stats import gaussian_kde
            xk = np.linspace(df[col].min(), df[col].max(), 200)
            ax.plot(xk, gaussian_kde(df[col])(xk), color='black', lw=2)
            ax.set_title(col); ax.set_xlabel('')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Pie + Donut":
        fig, axes = plt.subplots(1,2,figsize=(12,5))
        seg_counts = df['segment'].value_counts()
        colors_pie  = [C['gray'],C['blue'],C['amber'],C['violet']]
        axes[0].pie(seg_counts.values, labels=seg_counts.index,
                    colors=colors_pie, autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(edgecolor='white',lw=2))
        axes[0].set_title('Segment Distribution (Pie)')
        ch_counts = df['channel'].value_counts()
        axes[1].pie(ch_counts.values, labels=ch_counts.index,
                    colors=[C['blue'],C['teal'],C['green'],C['amber']],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(edgecolor='white',lw=2,width=0.55))
        axes[1].set_title('Channel Mix (Donut)')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

    elif chart == "Multi-Subplot Dashboard":
        fig = plt.figure(figsize=(16,10))
        gs = gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.38)
        fig.suptitle('Executive Dashboard', fontsize=14, fontweight='bold')
        seg_rev = df.groupby('segment')['revenue'].sum().sort_values(ascending=True)
        ax1 = fig.add_subplot(gs[0,0])
        ax1.barh(seg_rev.index, seg_rev.values,
                  color=[C['gray'],C['blue'],C['amber'],C['violet']],
                  alpha=0.85, edgecolor='white')
        ax1.set_title('Revenue by Segment')
        pivot2 = df.pivot_table('satisfaction','segment','channel',aggfunc='mean').round(2)
        ax2 = fig.add_subplot(gs[0,1])
        sns.heatmap(pivot2, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=3, vmax=5, ax=ax2, cbar=False, linewidths=0.5)
        ax2.set_title('Avg Satisfaction Heatmap')
        ax3 = fig.add_subplot(gs[0,2])
        ax3.scatter(df.sample(300,random_state=1)['annual_income'],
                    df.sample(300,random_state=1)['annual_spend'],
                    alpha=0.3, s=15, color=C['blue'])
        ax3.set_title('Income vs Spend')
        monthly2 = df.set_index('join_date').resample('ME')['revenue'].sum().reset_index()
        ax4 = fig.add_subplot(gs[1,:2])
        ax4.plot(range(len(monthly2)), monthly2['revenue'], color=C['teal'], lw=2)
        ax4.fill_between(range(len(monthly2)), monthly2['revenue'], alpha=0.2, color=C['teal'])
        ax4.set_title('Monthly Revenue Trend')
        ax5 = fig.add_subplot(gs[1,2])
        ch_rev = df.groupby('channel')['revenue'].sum()
        ax5.pie(ch_rev.values, labels=ch_rev.index, autopct='%1.0f%%',
                colors=[C['blue'],C['teal'],C['green'],C['amber']],
                wedgeprops=dict(edgecolor='white',lw=2,width=0.55))
        ax5.set_title('Revenue by Channel')
        plt.tight_layout(); st.pyplot(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 8 — CORRELATION MATRIX
# ════════════════════════════════════════════════════════════════════════════
elif page == "t8":
    st.title("🔗 Task 8 — Correlation Matrix Analysis")
    df = make_retail()
    num_cols = ['annual_income','annual_spend','num_orders','satisfaction',
                'tenure_years','loyalty_score','spend_rate','product_rating','return_rate']

    tab1,tab2,tab3,tab4 = st.tabs(["🗺️ Heatmaps","📐 Method Comparison","🧮 Significance","🔀 Clustering"])

    with tab1:
        method = st.radio("Correlation method", ["Pearson","Spearman","Kendall"], horizontal=True)
        method_map = {'Pearson':'pearson','Spearman':'spearman','Kendall':'kendall'}
        corr = df[num_cols].corr(method=method_map[method])
        mask = np.triu(np.ones_like(corr, dtype=bool))
        fig_h, ax_h = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                    vmin=-1, vmax=1, ax=ax_h, linewidths=0.5,
                    cbar_kws={'shrink':.75})
        ax_h.set_title(f'{method} Correlation Matrix', fontweight='bold', fontsize=13)
        plt.tight_layout(); st.pyplot(fig_h, use_container_width=True)

        target = st.selectbox("Target feature (bar chart)", num_cols)
        target_corr = corr[target].drop(target).sort_values()
        fig_b, ax_b = plt.subplots(figsize=(9,4))
        colors_b = [C['green'] if v>0 else C['red'] for v in target_corr.values]
        ax_b.barh(target_corr.index, target_corr.values,
                  color=colors_b, alpha=0.85, edgecolor='white')
        ax_b.axvline(0, color='black', lw=1)
        ax_b.set_title(f'{method} Correlation with {target}', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_b, use_container_width=True)

    with tab2:
        p_corr  = df[num_cols].corr('pearson').values
        sp_corr = df[num_cols].corr('spearman').values
        mask_m  = np.triu(np.ones_like(p_corr, dtype=bool))
        p_vals  = p_corr[~mask_m]
        sp_vals = sp_corr[~mask_m]
        fig_m, ax_m = plt.subplots(figsize=(7,6))
        ax_m.scatter(p_vals, sp_vals, alpha=0.55, s=50, color=C['blue'])
        ax_m.plot([-1,1],[-1,1], color=C['red'], lw=2, ls='--')
        r_agree = np.corrcoef(p_vals,sp_vals)[0,1]
        ax_m.set_xlabel('Pearson r'); ax_m.set_ylabel('Spearman ρ')
        ax_m.set_title(f'Pearson vs Spearman Agreement (r={r_agree:.4f})', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_m, use_container_width=True)

    with tab3:
        def pval_matrix(df_sub, method='pearson'):
            cols = df_sub.columns
            pvals = pd.DataFrame(np.ones((len(cols),len(cols))), columns=cols, index=cols)
            for i in range(len(cols)):
                for j in range(i+1,len(cols)):
                    if method=='pearson': r,p = pearsonr(df_sub.iloc[:,i], df_sub.iloc[:,j])
                    else:                  r,p = spearmanr(df_sub.iloc[:,i],df_sub.iloc[:,j])
                    pvals.iloc[i,j] = p; pvals.iloc[j,i] = p
            return pvals
        pvals = pval_matrix(df[num_cols])
        sig   = (pvals < 0.05).astype(int)
        fig_s, axes_s = plt.subplots(1,2,figsize=(14,6))
        neg_log_p = -np.log10(pvals.replace(0,1e-300))
        mask_s    = np.triu(np.ones_like(neg_log_p, dtype=bool))
        sns.heatmap(neg_log_p, mask=mask_s, annot=True, fmt='.1f',
                    cmap='Blues', ax=axes_s[0], linewidths=0.5,
                    cbar_kws={'label':'-log₁₀(p)','shrink':.75})
        axes_s[0].set_title('-log₁₀(p) Significance Matrix', fontweight='bold')
        sns.heatmap(sig, mask=mask_s, annot=True, fmt='d',
                    cmap=['#FEE2E2','#D1FAE5'], ax=axes_s[1], linewidths=0.5, cbar=False)
        axes_s[1].set_title('Significant Pairs (1 = p<0.05)', fontweight='bold')
        plt.tight_layout(); st.pyplot(fig_s, use_container_width=True)
        st.success(f"✅ {sig.values[~mask_s].sum()//2} / {len(num_cols)*(len(num_cols)-1)//2} pairs significant at α=0.05")

    with tab4:
        corr_c = df[num_cols].corr()
        fig_cl = sns.clustermap(corr_c, method='ward', cmap='RdBu_r',
                                vmin=-1, vmax=1, figsize=(9,8),
                                annot=True, fmt='.2f',
                                cbar_kws={'shrink':.6})
        fig_cl.fig.suptitle('Hierarchical Clustered Correlation Matrix', y=1.01, fontweight='bold')
        st.pyplot(fig_cl.fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TASK 9 — PANDAS DATA CLEANING
# ════════════════════════════════════════════════════════════════════════════
elif page == "t9":
    st.title("🧼 Task 9 — Pandas Data Cleaning")
    df_clean = make_retail()

    tab1,tab2,tab3 = st.tabs(["🔍 Data Quality","⚙️ Cleaning Steps","✅ Validation"])

    with tab1:
        st.subheader("Dataset Overview")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Rows", f"{len(df_clean):,}")
        c2.metric("Columns", len(df_clean.columns))
        c3.metric("Missing Cells", df_clean.isnull().sum().sum())
        c4.metric("Duplicates", df_clean.duplicated().sum())

        st.subheader("Data Types")
        dtype_df = pd.DataFrame({'Column':df_clean.columns,
                                  'Dtype':df_clean.dtypes.astype(str).values,
                                  'Non-null':df_clean.notnull().sum().values,
                                  'Unique':df_clean.nunique().values})
        st.dataframe(dtype_df, use_container_width=True, height=400)

    with tab2:
        step = st.selectbox("Explore cleaning step", [
            "1. Column Standardisation",
            "2. Type Conversion",
            "3. Missing Value Treatment",
            "4. Duplicate Removal",
            "5. Invalid Value Repair",
            "6. Categorical Standardisation",
            "7. Outlier Winsorisation",
            "8. Feature Engineering",
            "9. Categorical Encoding",
        ])

        if "Column" in step:
            st.markdown("### Column Name Standardisation")
            st.markdown("**Process:** `str.strip()` → `str.lower()` → `str.replace(' ','_')`")
            st.code("""# Pandas one-liner
df.columns = (df.columns
               .str.strip()
               .str.lower()
               .str.replace(r'\\s+', '_', regex=True))""", language='python')

        elif "Type" in step:
            st.markdown("### Data Type Conversion")
            st.code("""# Convert mixed-format strings to numeric/datetime
df['annual_income'] = (df['annual_income'].astype(str)
    .str.replace(r'[\\$,]', '', regex=True)
    .pipe(pd.to_numeric, errors='coerce'))

df['join_date'] = pd.to_datetime(df['join_date'],
    infer_datetime_format=True, errors='coerce')

bool_map = {'yes':1,'y':1,'no':0,'n':0}
df['discount_used'] = df['discount_used'].str.lower().map(bool_map)""", language='python')

        elif "Missing" in step:
            st.markdown("### Missing Value Strategies")
            strategies = pd.DataFrame({
                'Column':['annual_income','annual_spend','satisfaction','tenure_years','gender','category'],
                'Strategy':['Median per segment','Median per segment','Mode','Global median',
                             'Fill "Unknown"','Mode'],
                'Reason':['MAR — depends on segment','MAR','Ordinal — mode appropriate',
                           'MCAR','No meaningful imputation','MCAR'],
            })
            st.dataframe(strategies, use_container_width=True)
            st.code("""# Group-aware imputation
df['annual_income'] = df.groupby('segment')['annual_income'].transform(
    lambda x: x.fillna(x.median()))""", language='python')

        elif "Duplicate" in step:
            st.markdown("### Duplicate Removal")
            st.code("""# Three levels of deduplication
df.drop_duplicates(inplace=True)                              # exact dups
df.drop_duplicates(subset=['customer_id'], keep='first', inplace=True)  # key dups
df.drop_duplicates(subset=['email','full_name'], keep='first', inplace=True)  # near-dups""",
                    language='python')

        elif "Invalid" in step:
            st.markdown("### Invalid Value Repair")
            st.code("""# Range-based validation + repair
df.loc[(df['age'] < 18) | (df['age'] > 100), 'age'] = np.nan
df['age'] = df['age'].fillna(df['age'].median()).round().astype(int)

# Email validation via regex
email_re = r'^[\\w.+\\-]+@[\\w\\-]+\\.[a-z]{2,}$'
df['email_valid'] = df['email'].str.match(email_re, na=False)
df.loc[~df['email_valid'], 'email'] = np.nan""", language='python')

        elif "Categorical" in step:
            st.markdown("### Categorical Standardisation")
            before = ['male','MALE','m','mal','Male ']
            after  = ['Male','Male','Male','Male','Male']
            st.dataframe(pd.DataFrame({'Before':before,'After':after}), use_container_width=True)
            st.code("""gender_norm = {'male':'Male','m':'Male','mal':'Male',
               'female':'Female','f':'Female'}
df['gender'] = df['gender'].str.lower().str.strip().map(gender_norm)
df['segment'] = pd.Categorical(df['segment'],
    categories=['Bronze','Silver','Gold','Platinum'], ordered=True)""", language='python')

        elif "Outlier" in step:
            st.markdown("### Outlier Detection & Winsorisation")
            num_c = 'annual_income'
            Q1,Q3 = df_clean[num_c].quantile(.25), df_clean[num_c].quantile(.75)
            IQR = Q3-Q1
            st.info(f"**{num_c}:** Q1={Q1:,.0f} | Q3={Q3:,.0f} | IQR={IQR:,.0f} | Fences=[{Q1-1.5*IQR:,.0f}, {Q3+1.5*IQR:,.0f}]")
            st.code("""# IQR Winsorisation — clip extremes, no rows deleted
Q1, Q3 = df[col].quantile(.25), df[col].quantile(.75)
IQR    = Q3 - Q1
df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)""", language='python')

        elif "Engineering" in step:
            st.markdown("### Feature Engineering with `.assign()`")
            df_fe = df_clean.assign(
                avg_order_value=lambda x: (x['annual_spend']/(x['num_orders']+1)).round(2),
                spend_rate=lambda x: (x['annual_spend']/x['annual_income']*100).round(3),
                loyalty_score=lambda x: (x['tenure_years']*x['num_orders']/10).round(2),
                is_high_value=lambda x: (x['annual_spend']>=x['annual_spend'].quantile(.75)).astype(int),
            )
            st.dataframe(df_fe[['avg_order_value','spend_rate','loyalty_score','is_high_value']].head(8),
                         use_container_width=True)

        elif "Encoding" in step:
            st.markdown("### Categorical Encoding")
            st.code("""# Ordinal encoding
seg_ord = {'Bronze':1,'Silver':2,'Gold':3,'Platinum':4}
df['segment_code'] = df['segment'].map(seg_ord)

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender_code'] = le.fit_transform(df['gender'])

# One-hot encoding
ohe = pd.get_dummies(df['region'], prefix='rgn', dtype=int)
df  = pd.concat([df, ohe], axis=1)

# Frequency encoding
cat_freq = df['category'].value_counts(normalize=True)
df['category_freq'] = df['category'].map(cat_freq)""", language='python')

    with tab3:
        st.subheader("Validation Suite")
        checks = [
            ('No missing values in core cols',
             df_clean[['annual_income','annual_spend','num_orders','satisfaction']].isnull().sum().sum()==0,
             'null_count=0'),
            ('No exact duplicates',
             not df_clean.duplicated().any(), f'dups={df_clean.duplicated().sum()}'),
            ('Age in [18,100]',
             ((df_clean['age']>=18)&(df_clean['age']<=100)).all(),
             f'range=[{df_clean["age"].min()},{df_clean["age"].max()}]'),
            ('Satisfaction in [1,5]',
             ((df_clean['satisfaction']>=1)&(df_clean['satisfaction']<=5)).all(),
             f'range=[{df_clean["satisfaction"].min()},{df_clean["satisfaction"].max()}]'),
            ('Annual income > 0',
             (df_clean['annual_income']>0).all(), f'min={df_clean["annual_income"].min():.0f}'),
            ('Annual spend > 0',
             (df_clean['annual_spend']>0).all(),  f'min={df_clean["annual_spend"].min():.0f}'),
            ('Segment has 4 valid categories',
             df_clean['segment'].nunique()==4, str(df_clean['segment'].unique().tolist())),
            ('join_date is datetime',
             pd.api.types.is_datetime64_any_dtype(df_clean['join_date']), str(df_clean['join_date'].dtype)),
        ]
        results_df = pd.DataFrame([
            {'Check':name, 'Status':'✅ PASS' if cond else '❌ FAIL', 'Detail':detail}
            for name,cond,detail in checks
        ])
        st.dataframe(results_df, use_container_width=True)
        n_pass = sum(1 for _,c,_ in checks if c)
        st.success(f"✅ {n_pass}/{len(checks)} checks passed")

# ════════════════════════════════════════════════════════════════════════════
# TASK 10 — DATA MANIPULATION
# ════════════════════════════════════════════════════════════════════════════
elif page == "t10":
    st.title("🐼 Task 10 — Pandas Data Manipulation")
    df = make_retail()

    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "🔎 Filter & Sort","🔢 Aggregation","➕ Calculations",
        "🔄 Reshape & Windows","💼 Business Questions"])

    with tab1:
        st.subheader("Interactive Filtering")
        c1,c2,c3 = st.columns(3)
        seg_f  = c1.multiselect("Segment", ['Bronze','Silver','Gold','Platinum'],
                                 default=['Gold','Platinum'])
        reg_f  = c2.multiselect("Region",  ['North','South','East','West'],
                                 default=['North','South','East','West'])
        min_sp = c3.slider("Min annual spend ($)",
                            int(df['annual_spend'].min()), int(df['annual_spend'].max()),
                            int(df['annual_spend'].quantile(.25)))

        filtered = df[df['segment'].isin(seg_f) &
                      df['region'].isin(reg_f) &
                      (df['annual_spend'] >= min_sp)]
        st.info(f"Filtered: **{len(filtered):,}** of {len(df):,} rows")
        st.dataframe(filtered[['customer_id','segment','region','annual_income',
                                'annual_spend','num_orders','satisfaction']].head(20),
                     use_container_width=True)

        st.subheader("Sorting")
        sort_col = st.selectbox("Sort by", ['annual_spend','annual_income','num_orders','satisfaction'])
        sort_asc = st.radio("Order", ["Descending","Ascending"], horizontal=True) == "Ascending"
        st.dataframe(filtered.sort_values(sort_col, ascending=sort_asc)
                     [['customer_id','segment','annual_income','annual_spend','num_orders']]
                     .head(10), use_container_width=True)

    with tab2:
        st.subheader("GroupBy Aggregation")
        group_col = st.selectbox("Group by", ['segment','region','category','channel'])
        agg_func  = st.multiselect("Aggregations",
                                   ['mean','sum','count','std','median'],
                                   default=['mean','count'])
        if agg_func:
            agg_result = df.groupby(group_col)['annual_spend'].agg(agg_func).round(2)
            st.dataframe(agg_result, use_container_width=True)

        st.subheader("Pivot Table")
        pivot_val = st.selectbox("Value", ['annual_spend','revenue','satisfaction','num_orders'])
        pivot_row = st.selectbox("Row (index)", ['segment','region'])
        pivot_col = st.selectbox("Column", ['channel','category','region','segment'],
                                 index=0 if pivot_row=='segment' else 1)
        if pivot_row != pivot_col:
            piv = df.pivot_table(pivot_val, pivot_row, pivot_col, aggfunc='mean').round(2)
            st.dataframe(piv, use_container_width=True)
            fig_pv, ax_pv = plt.subplots(figsize=(9,4))
            sns.heatmap(piv, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax_pv,
                        linewidths=0.5, cbar_kws={'shrink':.75})
            ax_pv.set_title(f'Avg {pivot_val} — {pivot_row} × {pivot_col}', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_pv, use_container_width=True)

    with tab3:
        st.subheader("Calculated Columns")
        df_calc = df.assign(
            gross_revenue   = lambda x: x['unit_price']*x['quantity'],
            discount_amount = lambda x: (x['gross_revenue']*x['discount_pct']/100).round(2)
                              if 'gross_revenue' in x.columns
                              else (x['unit_price']*x['quantity']*x['discount_pct']/100).round(2),
            net_revenue     = lambda x: (x['unit_price']*x['quantity']*
                                         (1-x['discount_pct']/100)).round(2),
            spend_rate      = lambda x: (x['annual_spend']/x['annual_income']*100).round(3),
            loyalty_score   = lambda x: (x['tenure_years']*x['num_orders']/10).round(2),
            income_norm     = lambda x: ((x['annual_income']-x['annual_income'].min())/
                                          (x['annual_income'].max()-x['annual_income'].min())).round(4),
            spend_zscore    = lambda x: ((x['annual_spend']-x['annual_spend'].mean())/
                                          x['annual_spend'].std()).round(4),
        )
        show_cols = ['customer_id','segment','annual_income','annual_spend',
                     'net_revenue','spend_rate','loyalty_score','income_norm','spend_zscore']
        st.dataframe(df_calc[show_cols].head(10), use_container_width=True)

        st.subheader("Binning")
        df_bin = df.copy()
        df_bin['age_group']   = pd.cut(df_bin['age'],bins=[17,25,35,45,55,100],
                                        labels=['18-25','26-35','36-45','46-55','55+'])
        df_bin['spend_quartile']= pd.qcut(df_bin['annual_spend'],q=4,
                                           labels=['Q1 Low','Q2 Med-Low','Q3 Med-High','Q4 High'])
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Age Groups (pd.cut)**")
            st.dataframe(df_bin['age_group'].value_counts().sort_index()
                         .reset_index().rename(columns={'index':'age_group','age_group':'count'}),
                         use_container_width=True)
        with c2:
            st.markdown("**Spend Quartiles (pd.qcut)**")
            st.dataframe(df_bin['spend_quartile'].value_counts().sort_index()
                         .reset_index().rename(columns={'index':'spend_quartile','spend_quartile':'count'}),
                         use_container_width=True)

    with tab4:
        st.subheader("Melt — Wide to Long")
        seg_wide = df.groupby('segment').agg(
            avg_income=('annual_income','mean'),
            avg_spend =('annual_spend', 'mean'),
            avg_orders=('num_orders',   'mean'),
        ).round(2).reset_index()
        seg_long = seg_wide.melt(id_vars='segment',
                                  var_name='metric', value_name='value')
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Wide format**"); st.dataframe(seg_wide, use_container_width=True)
        with c2:
            st.markdown("**Long format**"); st.dataframe(seg_long.head(8), use_container_width=True)

        st.subheader("Rolling Windows")
        monthly = (df.set_index('join_date').resample('ME')['annual_spend']
                     .mean().reset_index().sort_values('join_date'))
        monthly['ma3'] = monthly['annual_spend'].rolling(3,min_periods=1).mean().round(2)
        monthly['ma6'] = monthly['annual_spend'].rolling(6,min_periods=1).mean().round(2)
        monthly['mom'] = monthly['annual_spend'].pct_change()*100

        fig_w, axes_w = plt.subplots(2,1,figsize=(12,7),sharex=True)
        axes_w[0].plot(range(len(monthly)), monthly['annual_spend'],
                       color=C['blue'], lw=1.8, alpha=0.7, label='Monthly')
        axes_w[0].plot(range(len(monthly)), monthly['ma3'],
                       color=C['teal'], lw=2.5, label='3M MA')
        axes_w[0].plot(range(len(monthly)), monthly['ma6'],
                       color=C['orange'], lw=2.5, ls='--', label='6M MA')
        axes_w[0].set_ylabel('Avg Spend ($)'); axes_w[0].legend(fontsize=9)
        axes_w[0].set_title('Rolling Window Functions', fontweight='bold')
        colors_mom = [C['green'] if v>=0 else C['red'] for v in monthly['mom'].fillna(0)]
        axes_w[1].bar(range(len(monthly)), monthly['mom'].fillna(0),
                      color=colors_mom, alpha=0.78, edgecolor='white')
        axes_w[1].axhline(0, color='black', lw=1)
        axes_w[1].set_ylabel('MoM Change (%)')
        axes_w[1].set_title('Month-over-Month Change')
        plt.tight_layout(); st.pyplot(fig_w, use_container_width=True)

    with tab5:
        q_choice = st.selectbox("Select business question", [
            "Q1: Revenue by segment",
            "Q2: Discount impact on spend",
            "Q3: Top 10 highest-value customers",
            "Q4: Satisfaction by segment × channel",
            "Q5: Monthly revenue trend",
            "Q6: Category performance",
            "Q7: Cohort analysis by join year",
        ])

        if "Q1" in q_choice:
            result = df.groupby('segment').agg(
                customers=('customer_id','count'),
                total_revenue=('revenue','sum'),
                avg_spend=('annual_spend','mean')
            ).sort_values('total_revenue',ascending=False).round(2)
            result['revenue_share%'] = (result['total_revenue']/result['total_revenue'].sum()*100).round(1)
            st.dataframe(result, use_container_width=True)
            fig_q,ax_q = plt.subplots(figsize=(8,4))
            ax_q.bar(result.index, result['total_revenue'],
                     color=[C['gray'],C['blue'],C['amber'],C['violet']],
                     alpha=0.85, edgecolor='white')
            ax_q.set_title('Total Revenue by Segment', fontweight='bold')
            ax_q.set_ylabel('Total Revenue ($)')
            plt.tight_layout(); st.pyplot(fig_q, use_container_width=True)

        elif "Q2" in q_choice:
            result = df.groupby('discount_used')['annual_spend'].agg(
                ['mean','median','count','std']).round(2)
            result.index = ['No Discount','Used Discount']
            st.dataframe(result, use_container_width=True)
            lift = (result.loc['Used Discount','mean']/result.loc['No Discount','mean']-1)*100
            st.metric("Spend lift from discounts", f"+{lift:.1f}%")

        elif "Q3" in q_choice:
            df['income_norm']  = (df['annual_income']-df['annual_income'].min())/(df['annual_income'].max()-df['annual_income'].min())
            df['spend_norm']   = (df['annual_spend'] -df['annual_spend'].min()) /(df['annual_spend'].max() -df['annual_spend'].min())
            df['orders_norm']  = (df['num_orders']   -df['num_orders'].min())   /(df['num_orders'].max()   -df['num_orders'].min())
            df['composite']    = (df['income_norm']*.3+df['spend_norm']*.4+df['orders_norm']*.3).round(4)
            top10 = df.nlargest(10,'composite')[
                ['customer_id','segment','annual_income','annual_spend','num_orders','composite']]
            st.dataframe(top10, use_container_width=True)

        elif "Q4" in q_choice:
            pivot_q4 = df.pivot_table('satisfaction','segment','channel',aggfunc='mean').round(3)
            pivot_q4['Overall'] = df.groupby('segment')['satisfaction'].mean().round(3)
            st.dataframe(pivot_q4, use_container_width=True)
            fig_q4, ax_q4 = plt.subplots(figsize=(8,5))
            sns.heatmap(pivot_q4, annot=True, fmt='.2f', cmap='RdYlGn',
                        vmin=3, vmax=5, ax=ax_q4, linewidths=0.5)
            ax_q4.set_title('Avg Satisfaction — Segment × Channel', fontweight='bold')
            plt.tight_layout(); st.pyplot(fig_q4, use_container_width=True)

        elif "Q5" in q_choice:
            monthly_rev = (df.set_index('join_date').resample('ME')['revenue']
                             .sum().reset_index())
            fig_q5,ax_q5 = plt.subplots(figsize=(12,4))
            ax_q5.plot(range(len(monthly_rev)), monthly_rev['revenue'],
                       color=C['teal'], lw=2.5)
            ax_q5.fill_between(range(len(monthly_rev)), monthly_rev['revenue'],
                                alpha=0.18, color=C['teal'])
            ax_q5.set_title('Monthly Total Revenue', fontweight='bold')
            ax_q5.set_ylabel('Revenue ($)')
            plt.tight_layout(); st.pyplot(fig_q5, use_container_width=True)

        elif "Q6" in q_choice:
            cat_perf = df.groupby('category').agg(
                customers=('customer_id','count'),
                avg_spend=('annual_spend','mean'),
                avg_rating=('product_rating','mean'),
                total_revenue=('revenue','sum'),
            ).round(2).sort_values('total_revenue',ascending=False)
            st.dataframe(cat_perf, use_container_width=True)

        elif "Q7" in q_choice:
            cohort = df.groupby('join_year').agg(
                customers=('customer_id','count'),
                avg_income=('annual_income','mean'),
                avg_spend=('annual_spend','mean'),
                avg_orders=('num_orders','mean'),
            ).round(2)
            st.dataframe(cohort, use_container_width=True)
            fig_q7,ax_q7 = plt.subplots(figsize=(8,4))
            ax_q7.bar(cohort.index, cohort['avg_spend'],
                      color=[C['blue'],C['teal'],C['green'],C['amber'],C['violet']],
                      alpha=0.85, edgecolor='white', width=0.6)
            ax_q7.set_title('Avg Spend by Join-Year Cohort', fontweight='bold')
            ax_q7.set_ylabel('Avg Annual Spend ($)')
            plt.tight_layout(); st.pyplot(fig_q7, use_container_width=True)
