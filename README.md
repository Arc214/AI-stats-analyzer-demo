# 🔬 Smart‑Stats V2 • AI‑Powered Statistical Analyzer

A Streamlit web app that lets anyone upload their own dataset (CSV/Excel) and automatically generates, edits, and runs a suite of informative statistical tests using a local LLM (Flan‑T5). No data‑cleaning step—just upload and analyze.

---

## 🚀 Key Features

* **Bring‑Your‑Own‑Data**: Easily upload CSV or Excel files via sidebar.
* **AI‑Driven Test Selection**: Local LLM suggests up to 4 relevant tests (correlation, t‑test, ANOVA, chi‑square, linear/logistic regression) based on your schema.
* **Editable Plan**: Review and tweak the JSON analysis plan before running.
* **Interactive Visualizations**: Scatter plots, boxplots, heatmaps, regression summaries, and confusion matrices with Plotly & Matplotlib.
* **Plain‑English Summary**: LLM‑generated narrative interpretation of results.

---

## 📋 Requirements

* Python 3.8+
* Packages listed in `requirements.txt`:

  ```text
  streamlit>=1.35
  pandas>=2.2
  numpy
  scipy<=1.15.3
  statsmodels==0.14.4
  scikit-learn
  matplotlib
  seaborn
  plotly
  transformers>=4.41
  torch
  sentencepiece
  ```

---

## 🔧 Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/your‑org/smart‑stats‑v2.git
   cd smart‑stats‑v2
   ```
2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Usage

1. **Run the app**

   ```bash
   streamlit run smart_stats_v2_no_cleaning.py
   ```
2. **Upload your dataset** via the sidebar (CSV or Excel).
3. **Review the AI‑generated JSON analysis plan**, editing column names or test types as needed.
4. **Click “Run Analyses”** to execute tests and view results.
5. **Read the AI‑generated summary** to understand key insights in plain English.

---

## 📝 Analysis Plan Format

Paste a JSON array of test specifications into the editor; for example:

```json
[
  {
    "name": "Correlation: Temp vs Usage",
    "type": "correlation",
    "cols": ["High Temp (°F)", "Total"]
  },
  {
    "name": "ANOVA: Usage by Weekday",
    "type": "anova_by_category",
    "numeric_col": "Total",
    "categorical_col": "Day"
  }
]
```

* **type** must be one of: `correlation`, `t_test_by_category`, `anova_by_category`, `chi_square`, `linear_regression`, `logistic_regression`.
* Field names vary slightly by test (see example).

---

## 🤝 Contributing

1. Fork the repo.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to branch (`git push origin feature-name`).
5. Open a pull request.

---

## 📄 License

This project is open‑source under the MIT License. See [LICENSE](LICENSE) for details.

---

*Made with ❤️ & open‑source models*
