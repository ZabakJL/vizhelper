from IPython.display import display, HTML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, recall_score
import textwrap
from scipy.stats import gaussian_kde

def show_histograms(df, name="DataFrame", n_cols=3, show_kde=True, height=3, bins=None):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("⚠️ No numeric variables found to plot.")
        return

    display(HTML(f"<h4>📈 Histograms – <code>{name}</code></h4>"))
    
    n_vars = len(numeric_df.columns)
    n_rows = (n_vars + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * height))
    axes = axes.flatten()

    for i, col in enumerate(numeric_df.columns):
        ax = axes[i]
        ax.hist(numeric_df[col].dropna(), bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_title(col)
        if show_kde:
            x_vals = np.linspace(numeric_df[col].min(), numeric_df[col].max(), 200)
            ax2 = ax.twinx()
            try:
                kde = gaussian_kde(numeric_df[col].dropna())
                ax2.plot(x_vals, kde(x_vals), color='red', linewidth=1)
            except:
                pass
            ax2.set_yticks([])

    # Eliminar ejes sobrantes si hay
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def show_correlogram(df, name="DataFrame", hue=None, diag_kind="hist"):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("⚠️ No numeric variables found for correlogram.")
        return
    plot_df = pd.concat([numeric_df, df[hue]], axis=1) if hue and hue in df.columns else numeric_df
    display(HTML(f"<h4>🔗 Correlogram – <code>{name}</code></h4>"))
    g = sns.PairGrid(plot_df, hue=hue, diag_sharey=False)
    g.map_diag(sns.histplot if diag_kind == "hist" else sns.kdeplot)
    g.map_offdiag(sns.scatterplot)
    if hue:
        g.add_legend()
    plt.tight_layout()
    plt.show()

def show_correlation_matrix(df, name="DataFrame", method="pearson", annot=True, fmt=".2f", cmap="coolwarm", view="both"):
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        print("⚠️ No numeric variables found for correlation matrix.")
        return
    corr = numeric_df.corr(method=method)
    display(HTML(f"<h4>🧮 Correlation Matrix – <code>{name}</code></h4>"))
    if view in ["table", "both"]:
        display(corr.style.format(precision=6))
    if view in ["heatmap", "both"]:
        #plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5, square=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

def show_strong_correlations(df, threshold=0.7):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("⚠️ No numeric variables found.")
        return
    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    strong_pairs = upper.stack().reset_index()
    strong_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
    strong_pairs = strong_pairs[strong_pairs['Correlation'] >= threshold].sort_values(by='Correlation', ascending=False)
    display(strong_pairs)

def show_top_correlograms(df, threshold=0.75, max_plots=12, n_cols=3, height=3):
    """
    Muestra los scatterplots de los pares de variables numéricas con correlación mayor al threshold.
    
    Parámetros:
    - df: DataFrame original.
    - threshold: Umbral mínimo de correlación para considerar el par.
    - max_plots: Número máximo de pares a graficar.
    - n_cols: Número de columnas en la grilla de plots.
    - height: Altura por subplot (ancho se ajusta automáticamente).
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("⚠️ No hay columnas numéricas en el DataFrame.")
        return
    
    # Calcular la matriz de correlación absoluta
    corr = numeric_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Filtrar pares con correlación fuerte
    strong_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={0: "Correlation", "level_0": "Variable 1", "level_1": "Variable 2"})
    )
    strong_pairs = strong_pairs[strong_pairs["Correlation"] >= threshold]
    strong_pairs = strong_pairs.sort_values(by="Correlation", ascending=False).head(max_plots)

    if strong_pairs.empty:
        print(f"ℹ️ No se encontraron pares con correlación mayor a {threshold}.")
        return

    # Título
    display(HTML(f"<h4>🔗 Correlogramas – Correlaciones mayores a {threshold}</h4>"))

    # Crear figura y ejes
    n_rows = (len(strong_pairs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4.5, n_rows * height))
    axes = axes.flatten()

    for i in range(len(strong_pairs)):
        ax = axes[i]
        x = strong_pairs.iloc[i]["Variable 1"]
        y = strong_pairs.iloc[i]["Variable 2"]
        corr_val = strong_pairs.iloc[i]["Correlation"]

        sns.scatterplot(data=numeric_df, x=x, y=y, ax=ax, s=20, alpha=0.7)

        # Título con wrap
        title = f"{x} vs {y} (r={corr_val:.2f})"
        wrapped_title = "\n".join(textwrap.wrap(title, width=30))
        ax.set_title(wrapped_title, fontsize=9)

        # Etiquetas de ejes con rotación y ajuste
        ax.set_xlabel(x, rotation=0, ha='center', fontsize=8)
        ax.set_ylabel(y, fontsize=8)

    # Eliminar ejes vacíos
    for j in range(len(strong_pairs), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.show()

def plot_classification_metrics(y_true, y_pred, y_prob):
    """
    Compute and visualize classification performance metrics for a predictive model.

    This function calculates multiple performance metrics for classification models
    such as F1 Score, Accuracy, Error Rate, Precision, Sensitivity (Recall), Specificity,
    and Area Under the ROC Curve (AUC). It also displays three visualizations:
    - ROC curve with AUC
    - Confusion matrix heatmap
    - Horizontal bar chart comparing metric values

    Parameters:
    -----------
    y_true : array-like
        Ground truth (true) class labels.
    
    y_pred : array-like
        Predicted class labels from the model.
    
    y_prob : array-like
        Predicted probability scores for the positive class (used to compute ROC and AUC).
        For binary classification, this should be a 1D or 2D array with probabilities.

    Returns:
    --------
    Tuple[List[float], List[str]]
        A tuple containing:
        - A list of metric values in the order: 
          F1 Score, Accuracy, Error Rate, Precision, Sensitivity, Specificity, AUC
          (metrics may vary depending on the number of classes)
        - A list of corresponding metric names

    Notes:
    ------
    For multiclass problems, Sensitivity and Specificity are computed as macro averages.
    If AUC cannot be calculated (e.g., due to input shape), its value is set to None.
    """
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    metrics_dict = {}

    metrics_dict['F1 Score'] = metrics.f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    metrics_dict['Accuracy'] = metrics.accuracy_score(y_true=y_true, y_pred=y_pred)
    metrics_dict['Error Rate'] = 1 - metrics_dict['Accuracy']
    metrics_dict['Precision'] = metrics.precision_score(y_true=y_true, y_pred=y_pred, average='macro')

    if len(conf_matrix) == 2:
        TN, FP, FN, TP = conf_matrix.ravel()
        metrics_dict['Sensitivity'] = TP / (TP + FN)
        metrics_dict['Specificity'] = TN / (TN + FP)
    else:
        print("Classification Report:")
        print(classification_report(y_true, y_pred))
        metrics_dict['Specificity'] = recall_score(y_true=y_true, y_pred=y_pred, average='macro')

    fpr, tpr, _ = metrics.roc_curve(y_true, y_prob.ravel(), pos_label=1)
    try:
        metrics_dict['AUC'] = metrics.auc(fpr, tpr)
    except:
        metrics_dict['AUC'] = None

    sns.set(font_scale=0.8)
    fig = plt.figure(figsize=(10, 3), dpi=300)
    gs = gridspec.GridSpec(1, 3)

    ax = fig.add_subplot(gs[0:1])
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label=f'AUC = {metrics_dict["AUC"]:.2f}')
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([-0.01, 1])
    ax.set_ylim([0, 1.05])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    ax.set_facecolor("whitesmoke")

    ax = fig.add_subplot(gs[0, 1])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title('Confusion Matrix')
    if len(conf_matrix) == 2:
        class_labels = ['False', 'True']
        plt.xticks(ticks=np.arange(2) + 0.5, labels=class_labels)
        plt.yticks(ticks=np.arange(2) + 0.5, labels=class_labels)

    ax = fig.add_subplot(gs[0, 2])
    metric_names = list(metrics_dict.keys())
    metric_values = list(metrics_dict.values())
    bars = ax.barh(metric_names, metric_values, linewidth=0.8, color="skyblue", edgecolor="silver")
    ax.set_title('Metric Comparison')
    ax.set_xlabel("Value")
    ax.bar_label(bars, labels=[f'{val:.4f}' for val in metric_values], label_type='center')
    ax.set_facecolor("whitesmoke")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    plt.tight_layout()
    plt.show()

    return metrics_dict
