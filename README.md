# vizhelper

**Authors:** Karen Yorlady, Juan Sebastián, Juan Pablo

A Python package for customized and visually enhanced data exploration and classification metric evaluation. Tailored for data scientists and notebook workflows.

## Features
- 📊 Styled summaries of DataFrames
- 📈 Visualization of classification performance metrics

## Installation
```bash
pip install git+https://github.com/yourusername/vizhelper.git
```

## Usage
```python
from vizhelper import summarize_dataframe_structure, plot_classification_metrics

# Summarize DataFrame structure
display(summarize_dataframe_structure(df))

# Plot classification metrics
plot_classification_metrics(y_test, y_pred, y_prob)
```

## License
MIT
