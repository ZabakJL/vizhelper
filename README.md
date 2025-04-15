# vizhelper

**Author:** ZabakJL

`vizhelper` is a modular Python package designed to accelerate exploratory data analysis (EDA) and model evaluation. It provides visual summaries of DataFrames and diagnostics for classification models — ideal for data scientists working in Jupyter notebooks or similar environments.

---

## ✨ Features

- 📋 **Modular structure** for flexibility and clarity
- 📊 Styled summaries of DataFrame structure
- 🔍 Preview of rows (sample, head, tail)
- 🧮 Basic statistics and unique categorical values
- 📈 Histograms, correlograms, and correlation matrices
- 🧪 Visual evaluation of classification models (ROC, confusion matrix, metric bars)
- 🚀 `explore_dataframe()` one-liner to run a full EDA report

---

## 📦 Installation

You can install the package directly from GitHub using `pip`:

### Latest version (main branch)
```bash
pip install git+https://github.com/ZabakJL/vizhelper.git
```

### Specific version (e.g., v0.2.0)
```bash
pip install git+https://github.com/ZabakJL/vizhelper.git@v0.2.0
```

> Ensure you have `git` installed and internet access to reach GitHub.

---

## 🚀 Quick Start

```python
from vizhelper import explore_dataframe

# Run full EDA
explore_dataframe(df, name="Customers")
```

You can also import functions individually:

```python
from vizhelper import (
    show_dataframe_structure, preview_dataframe,
    show_basic_stats, show_unique_categorical_values,
    show_histograms_sns, show_correlogram, show_correlation_matrix,
    plot_classification_metrics
)

# Example: Display classification model metrics
plot_classification_metrics(y_test, y_pred, y_prob)
```

---

## 🧾 Version History

| Version | Description                                                             |
|---------|-------------------------------------------------------------------------|
| v0.2.0  | Modular refactor, new `explore_dataframe()`, and `plot_classification_metrics` |
| v0.1.0  | Initial release: summary of DataFrames and classification metrics       |

---

## 🧪 Requirements

- Python ≥ 3.7
- pandas
- seaborn
- matplotlib
- scikit-learn
- numpy

---

## 📘 Full Manual (Advanced Usage)

This section provides a more detailed overview of the modules, individual functions, and how to customize your EDA reports.

### 📦 Module Breakdown

- **preview.py**
  - `show_dataframe_structure(df, name='DataFrame')`
  - `preview_dataframe(df, name='DataFrame', method='sample', n=5)`

- **statistics.py**
  - `show_basic_stats(df, name='DataFrame', include='number')`
  - `show_unique_categorical_values(df, name='DataFrame')`

- **visualization.py**
  - `show_histograms(df, name='DataFrame', n_cols=3, show_kde=True, height=3)`
  - `show_correlogram(df, name='DataFrame', hue=None, diag_kind='hist')`
  - `show_correlation_matrix(df, name='DataFrame', method='pearson', view='both')`
  - `show_strong_correlations(df, threshold=0.7)`
  - `show_top_correlograms(df, threshold=0.75, max_plots=12, n_cols=3, height=3)`
  - `plot_classification_metrics(y_true, y_pred, y_prob)`

- **explorer.py**
  - `explore_dataframe(df, name='DataFrame', functions=None, function_kwargs=None)`

### 🧪 Customizing Function Calls

You can use `function_kwargs` to pass parameters to specific steps:

```python
explore_dataframe(
    df,
    name='Sales Dataset',
    functions=['histograms', 'correlation', 'classification_metrics'],
    function_kwargs={
        'histograms': {'n_cols': 4, 'show_kde': False},
        'correlation': {'method': 'spearman', 'view': 'heatmap'},
        'classification_metrics': {
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
    }
)
```

> All visual functions are optimized for interactive environments like Jupyter Notebooks.

## 📄 License

MIT

