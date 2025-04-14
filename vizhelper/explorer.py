
from .preview import show_dataframe_structure, preview_dataframe
from .statistics import show_basic_stats, show_unique_categorical_values
from .visualization import show_histograms_sns, show_correlogram, show_correlation_matrix

def explore_dataframe(df, name="DataFrame", functions=None):
    all_functions = {
        'structure': lambda: show_dataframe_structure(df, name),
        'preview': lambda: preview_dataframe(df, name),
        'categorical': lambda: show_unique_categorical_values(df, name),
        'stats': lambda: show_basic_stats(df, name),
        'histograms': lambda: show_histograms_sns(df, name),
        'correlogram': lambda: show_correlogram(df, name),
        'correlation': lambda: show_correlation_matrix(df, name)
    }
    if functions is None:
        functions = list(all_functions.keys())
    for func in functions:
        if func in all_functions:
            print(f"▶️ Running: {func}")
            all_functions[func]()
        else:
            print(f"⚠️ Unknown function: {func}")
