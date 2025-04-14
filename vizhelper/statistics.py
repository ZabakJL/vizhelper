
from IPython.display import display, HTML
import pandas as pd

def show_basic_stats(df, name="DataFrame", include="number"):
    stats = df.describe(include=include).T
    html = f"<h4>📊 Basic statistics for <code>{name}</code></h4>"
    html += stats.to_html(classes="table table-bordered table-striped", border=0)
    display(HTML(html))

def show_unique_categorical_values(df, name="DataFrame"):
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    rows = []
    for col in categorical_cols:
        unique_vals = df[col].unique()
        val_str = ", ".join(map(str, unique_vals))
        rows.append((col, len(unique_vals), val_str))
    result_df = pd.DataFrame(rows, columns=["Column", "Unique Values", "Values"])
    html = f"<h4>🔍 Unique categorical values in <code>{name}</code></h4>"
    html += result_df.to_html(index=False, escape=False, classes='table table-striped', border=0)
    display(HTML(html))
