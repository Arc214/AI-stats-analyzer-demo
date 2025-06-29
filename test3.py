# ---------------------------------------------------------------------------------
#  Smart-Stats Streamlit App — V2 (No Data Cleaning)
#  Bring-Your-Own-Data Statistical Analyzer with Local LLM
#---------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import json
import textwrap
import warnings

# --- Statistical & Plotting Libs ---
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# 🧠 LOAD LIGHTWEIGHT LOCAL LLM (CACHED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Downloading / loading local LLM…")
def load_llm():
    from transformers import pipeline
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        tokenizer="google/flan-t5-base",
        max_length=1024,
        device=-1,
    )

llm = load_llm()

def ask_llm(prompt: str, max_new_tokens: int = 512) -> str:
    response = llm(prompt, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
    return response[0]["generated_text"].strip()

# ─────────────────────────────────────────────────────────────
#  STREAMLIT PAGE CONFIG & INITIALIZATION
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="Smart-Stats V2", layout="wide")
st.title("🔬 Smart-Stats V2 • AI-Powered Statistical Analyzer")

# --- Initialize Session State ---
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
    st.session_state.analysis_plan_str = ""
    st.session_state.analysis_results = ""

# ─────────────────────────────────────────────────────────────
#  FILE UPLOAD & DATA LOADING
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload Data")
    st.markdown("Upload a CSV or Excel file for analysis.")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    with st.spinner("📖 Reading your file…"):
        try:
            if uploaded_file.name.endswith(("xlsx", "xls")):
                st.session_state.raw_df = pd.read_excel(uploaded_file)
            else:
                st.session_state.raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()
else:
    st.info("👋 Welcome! Upload a dataset using the sidebar to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────
#  AI-DRIVEN STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────
st.header("2. AI-Driven Statistical Analysis")

df = st.session_state.raw_df.copy()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

if not st.session_state.analysis_plan_str:
    with st.spinner("🤖 AI is selecting relevant statistical tests…"):
        schema_desc = (
            f"Numeric columns: {', '.join(numeric_cols)}\n"
            f"Categorical columns: {', '.join(categorical_cols)}"
        )
        prompt_tests = textwrap.dedent(f"""
        You are a statistics expert. Given the data schema, suggest up to 4 informative statistical tests.
        Output a valid JSON list of tests with types: correlation, t_test_by_category, anova_by_category,
        chi_square, linear_regression, logistic_regression.

        Schema:
        {schema_desc}

        JSON Plan:
        """
        )
        st.session_state.analysis_plan_str = ask_llm(prompt_tests)

st.info("🤖 Proposed analysis plan (editable).", icon="🤖")
st.session_state.analysis_plan_str = st.text_area(
    "**Analysis Plan (JSON)**",
    value=st.session_state.analysis_plan_str,
    height=300,
)

if st.button("📈 Run Analyses", type="primary"):
    try:
        analysis_plan = json.loads(st.session_state.analysis_plan_str)
        st.write("---")
        st.subheader("📊 Analysis Results")
        results_summary = ""

        for i, test in enumerate(analysis_plan):
            name = test.get("name", f"Test {i+1}")
            ttype = test.get("type")
            with st.expander(f"**▶️ {name}**", expanded=True):
                try:
                    # CORRELATION
                    if ttype == "correlation":
                        x = pd.to_numeric(df[test["cols"][0]], errors="coerce")
                        y = pd.to_numeric(df[test["cols"][1]], errors="coerce")
                        df_plot = pd.DataFrame({'x': x, 'y': y}).dropna()
                        r, p = stats.pearsonr(df_plot['x'], df_plot['y'])
                        st.write(f"r = {r:.3f}, p = {p:.3g}")
                        fig = px.scatter(df_plot, x='x', y='y', trendline="ols")
                        st.plotly_chart(fig, use_container_width=True)
                        results_summary += f"- {name}: r={r:.3f}, p={p:.3g}\n"

                    # T-TEST BY CATEGORY
                    elif ttype == "t_test_by_category":
                        num_col = test["numeric_col"]
                        cat_col = test["categorical_col"]
                        groups = df[cat_col].dropna().unique()
                        if len(groups) == 2:
                            g1 = df[df[cat_col]==groups[0]][num_col].dropna()
                            g2 = df[df[cat_col]==groups[1]][num_col].dropna()
                            t_stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                            st.write(f"t = {t_stat:.3f}, p = {p:.3g}")
                            fig = px.box(df, x=cat_col, y=num_col)
                            st.plotly_chart(fig, use_container_width=True)
                            results_summary += f"- {name}: t={t_stat:.3f}, p={p:.3g}\n"

                    # ANOVA BY CATEGORY
                    elif ttype == "anova_by_category":
                        num_col = test["numeric_col"]
                        cat_col = test["categorical_col"]
                        groups = [df[df[cat_col]==g][num_col].dropna() for g in df[cat_col].unique()]
                        f_stat, p = stats.f_oneway(*groups)
                        st.write(f"F = {f_stat:.3f}, p = {p:.3g}")
                        fig = px.box(df, x=cat_col, y=num_col)
                        st.plotly_chart(fig, use_container_width=True)
                        results_summary += f"- {name}: F={f_stat:.3f}, p={p:.3g}\n"

                    # CHI-SQUARE
                    elif ttype == "chi_square":
                        cols = test["cols"]
                        table = pd.crosstab(df[cols[0]], df[cols[1]])
                        chi2, p, _, _ = stats.chi2_contingency(table)
                        st.write(f"χ² = {chi2:.3f}, p = {p:.3g}")
                        st.dataframe(table)
                        fig = px.imshow(table, text_auto=True)
                        st.plotly_chart(fig, use_container_width=True)
                        results_summary += f"- {name}: χ²={chi2:.3f}, p={p:.3g}\n"

                    # LINEAR REGRESSION
                    elif ttype == "linear_regression":
                        X = pd.to_numeric(df[test["cols"][0]], errors="coerce")
                        y = pd.to_numeric(df[test["cols"][1]], errors="coerce")
                        df_lr = pd.DataFrame({'X': X, 'y': y}).dropna()
                        model = sm.OLS(df_lr['y'], sm.add_constant(df_lr['X'])).fit()
                        st.write(f"R² = {model.rsquared:.3f}")
                        st.text(str(model.summary()))
                        fig = px.scatter(df_lr, x='X', y='y', trendline="ols")
                        st.plotly_chart(fig, use_container_width=True)
                        results_summary += f"- {name}: R²={model.rsquared:.3f}\n"

                    # LOGISTIC REGRESSION
                    elif ttype == "logistic_regression":
                        features = test["feature_cols"]
                        target = test["target_col"]
                        df_log = df[features + [target]].dropna()
                        X_train, X_test, y_train, y_test = train_test_split(
                            df_log[features], df_log[target], test_size=0.3, random_state=42)
                        clf = LogisticRegression().fit(X_train, y_train)
                        preds = clf.predict(X_test)
                        acc = accuracy_score(y_test, preds)
                        st.write(f"Accuracy = {acc:.3f}")
                        cm = confusion_matrix(y_test, preds)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                        st.pyplot(fig)
                        results_summary += f"- {name}: Accuracy={acc:.3f}\n"

                    else:
                        st.warning(f"Unsupported test type: {ttype}")
                except Exception as e:
                    st.error(f"Error in {name}: {e}")

        st.session_state.analysis_results = results_summary

    except json.JSONDecodeError:
        st.error("Invalid JSON plan. Please correct it.")

# ─────────────────────────────────────────────────────────────
#  AI-GENERATED SUMMARY
# ─────────────────────────────────────────────────────────────
if st.session_state.analysis_results:
    st.header("3. AI-Generated Summary")
    with st.spinner("🤖 Writing summary…"):
        summary_prompt = (
            "Explain these results to a non-technical person: \n" + st.session_state.analysis_results
        )
        summary = ask_llm(summary_prompt, max_new_tokens=400)
        st.success(summary)

# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Made with ❤️ & open-source models • V2 (No Cleaning)")
