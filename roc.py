import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import numpy as np

# Example data
y_true = np.array([0, 0, 1, 1])
y_scores_model1 = np.array([0.1, 0.4, 0.35, 0.8])
y_scores_model2 = np.array([0.2, 0.3, 0.6, 0.85])

fpr1, tpr1, _ = roc_curve(y_true, y_scores_model1)
roc_auc1 = auc(fpr1, tpr1)

fpr2, tpr2, _ = roc_curve(y_true, y_scores_model2)
roc_auc2 = auc(fpr2, tpr2)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=fpr1, y=tpr1,
    mode='lines',
    name=f'Model 1 (AUC = {roc_auc1:.2f})',
    line=dict(width=2),
    hoverinfo='name+x+y',
    legendgroup="Model 1",
    customdata=[['Model 1']]*len(fpr1),
))

fig.add_trace(go.Scatter(
    x=fpr2, y=tpr2,
    mode='lines',
    name=f'Model 2 (AUC = {roc_auc2:.2f})',
    line=dict(width=2),
    hoverinfo='name+x+y',
    legendgroup="Model 2",
    customdata=[['Model 2']]*len(fpr2),
))

fig.update_layout(
    title='Interactive ROC Curve',
    xaxis_title='False Positive Rate',
    yaxis_title='True Positive Rate',
    hovermode='closest',
    clickmode='event+select'
)

fig.show()
