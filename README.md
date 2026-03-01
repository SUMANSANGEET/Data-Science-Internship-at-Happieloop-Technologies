# 🔬 Data Science Tasks 1–10 — Streamlit Dashboard

An interactive multi-page Streamlit application covering all 10 data science coursework tasks.

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Open in browser
Streamlit will automatically open `http://localhost:8501`

---

## 📚 Tasks Covered

| Task | Title | Features |
|------|-------|----------|
| **Task 1** | Statistical Analysis | Iris dataset · ANOVA · normality tests · 6 chart types |
| **Task 2** | Data Cleaning Pipeline | Dirty data viz · cleaning steps · before/after comparison |
| **Task 3** | Exploratory EDA | Univariate · bivariate · correlation heatmap |
| **Task 4** | Linear Regression | Simple/multiple · Ridge/Lasso · diagnostics |
| **Task 5** | Seaborn Visualisation | 8 chart families · pair plot · facet grids |
| **Task 6** | Hypothesis Testing | 6 test types · effect sizes · power analysis |
| **Task 7** | Matplotlib Visualisation | 8 chart types · GridSpec dashboard |
| **Task 8** | Correlation Matrix | Pearson/Spearman/Kendall · significance · clustering |
| **Task 9** | Pandas Data Cleaning | Step-by-step walkthrough · validation suite |
| **Task 10** | Data Manipulation | Filter/sort · groupby · reshape · business questions |

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this folder to a **GitHub repository**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `app.py` as the main file
4. Click **Deploy** — your app will be live in ~2 minutes!

---

## 📦 Project Structure

```
streamlit_app/
├── app.py              ← Main application (all 10 tasks)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 💡 Usage Tips

- Use the **sidebar** to navigate between tasks
- Each task has **interactive tabs** for different analysis views
- **Sliders, dropdowns, and multiselects** let you explore the data dynamically
- All charts update **instantly** when you change controls
- Data is **generated on first load** and cached — no external files needed
