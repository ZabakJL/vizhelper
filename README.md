# vizhelper

**Authors:** Karen Yorlady, Juan Sebastián, Juan Pablo

A Python package for customized and visually enhanced data exploration and classification metric evaluation. Tailored for data scientists and notebook workflows.

## Features
- 📊 Styled summaries of DataFrames
- 📈 Visualization of classification performance metrics

## Installation

You can install the package directly from GitHub using `pip`:

### Latest version (main branch)
```bash
pip install git+https://github.com/ZabakJL/vizhelper.git
```

### Specific version (e.g., v0.1.0)
```bash
pip install git+https://github.com/ZabakJL/vizhelper.git@v0.1.0
```

> Make sure you have `git` installed and internet access to reach GitHub.

## Usage

```python
from vizhelper import summarize_dataframe_structure, plot_classification_metrics

# Summarize DataFrame structure
display(summarize_dataframe_structure(df))

# Plot classification metrics
plot_classification_metrics(y_test, y_pred, y_prob)
```

## Version History

| Version | Description                                                |
|---------|------------------------------------------------------------|
| v0.1.0  | Initial release: summary of DataFrames and classification metrics |

## License
MIT
