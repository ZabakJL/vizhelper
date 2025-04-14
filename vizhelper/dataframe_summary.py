import pandas as pd
from pandas.io.formats.style import Styler

def summarize_dataframe_structure(df: pd.DataFrame) -> Styler:
    """
    Generate a styled summary table that mimics df.info() output
    but with enhanced formatting for Jupyter Notebooks.
    """
    summary = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Total Rows': len(df)
    })
    summary['% Non-Null'] = (summary['Non-Null Count'] / summary['Total Rows']) * 100

    styled_summary = (
        summary.style
        .bar(subset=['% Non-Null'], color='#5fba7d')
        .background_gradient(subset=['Non-Null Count'], cmap='Greens')
        .format({'% Non-Null': '{:.1f}%'})
        .set_caption("DataFrame Summary (styled df.info())")
    )

    return styled_summary
