
from IPython.display import display, HTML
import pandas as pd

def show_dataframe_structure(df: pd.DataFrame, name="DataFrame"):
    summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Total Rows': len(df)
    })
    summary['% Non-Null'] = (summary['Non-Null Count'] / summary['Total Rows']) * 100

    display(HTML(f"<h4>📋 Structure of <code>{name}</code></h4>"))
    styled = (
        summary.style
        .bar(subset=['% Non-Null'], color='#5fba7d')
        .background_gradient(subset=['Non-Null Count'], cmap='Greens')
        .format({'% Non-Null': '{:.1f}%'})
        .set_caption(f"📄 Summary Table – `{name}`")
    )
    display(styled)

def preview_dataframe(df, name='DataFrame', method='sample', n=5):
    display(HTML(f"<h4>📋 Preview of <code>{name}</code> using method: '{method}'</h4>"))
    if method == 'head':
        preview = df.head(n)
    elif method == 'tail':
        preview = df.tail(n)
    else:
        preview = df.sample(n)
    display(HTML(preview.to_html(index=True, classes='table table-striped', border=0)))
