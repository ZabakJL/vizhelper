
from .preview import show_dataframe_structure, preview_dataframe
from .statistics import show_basic_stats, show_unique_categorical_values
from .visualization import show_histograms, show_correlogram, show_correlation_matrix, show_strong_correlations, show_top_correlograms, plot_classification_metrics

def explore_dataframe(df, name="DataFrame", functions=None, function_kwargs=None):
    """
    Run a custom or full sequence of EDA visualizations and summaries.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to explore.
    name : str
        Display name used in titles and outputs.
    functions : list or None
        List of function keys to run. If None, all are executed.
    function_kwargs : dict or None
        Dictionary of keyword arguments per function, e.g.,
        {
            'histograms': {'n_cols': 2, 'height': 3},
            'correlation': {'method': 'spearman'}
        }
    """
    all_functions = {
        'structure': lambda: show_dataframe_structure(df, name),
        'preview': lambda: preview_dataframe(df, name),
        'categorical': lambda: show_unique_categorical_values(df, name),
        'stats': lambda: show_basic_stats(df, name),
        'histograms': lambda: show_histograms(df, name, **function_kwargs.get('histograms', {})),
        'correlogram': lambda: show_correlogram(df, name, **function_kwargs.get('correlogram', {})),
        'correlation': lambda: show_correlation_matrix(df, name, **function_kwargs.get('correlation', {})),
        'strong_corr': lambda: show_strong_correlations(df, **function_kwargs.get('strong_corr', {})),
        'top_correlograms': lambda: show_top_correlograms(df, **function_kwargs.get('top_correlograms', {})),
        'classification_metrics': lambda: plot_classification_metrics(
            **function_kwargs.get('classification_metrics', {})
        )
    }

    if functions is None:
        functions = list(all_functions.keys())
    if function_kwargs is None:
        function_kwargs = {}

    for func in functions:
        if func in all_functions:
            print(f"▶️ Running: {func}")
            all_functions[func]()
        else:
            print(f"⚠️ Unknown function: {func}")
