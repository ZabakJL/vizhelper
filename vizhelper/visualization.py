
from IPython.display import display, HTML
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, recall_score

def show_histograms_sns(df, name="DataFrame", n_cols=3, show_kde=True, height=3):
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        print("⚠️ No numeric variables found to plot.")
        return
    display(HTML(f"<h4>📈 Histograms – <code>{name}</code></h4>"))
    melted = numeric_df.melt()
    g = sns.FacetGrid(melted, col='variable', col_wrap=n_cols, sharex=False, sharey=False, height=height)
    g.map(sns.histplot, 'value', kde=show_kde)
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
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=annot, fmt=fmt, cmap=cmap, linewidths=0.5, square=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
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
