from sklearn.metrics import (RocCurveDisplay, roc_curve, auc, 
accuracy_score, precision_score, recall_score, f1_score, 
roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelBinarizer
import scanpy as sc
import squidpy as sq
import numpy as np
import matplotlib.pyplot as plt

import os
os.chdir("/lustre/scratch/krolha")

def load_from_pickle(filename):
    import pickle
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

data = load_from_pickle("./bachelor/results/bachelor_pred_6.pickle")
test_pred_probabilities = load_from_pickle("./bachelor/results/test_preds_bachelor_pred_6.pickle")
sq.pl.spatial_scatter(data, color=['joint_leiden_clusters', 'cluster_predictions'])
print(len(np.where(data.obs['joint_leiden_clusters']==data.obs['cluster_predictions'])[0])/len(data.obs))

# Saving joint_leiden_clusters and cluster_predictions as y_true and y_pred
y_true = data.obs.joint_leiden_clusters.values.tolist()
y_pred = data.obs.cluster_predictions.values.tolist()

# Number of clusters
n_clusters = data.obs['joint_leiden_clusters'].unique().shape[0]

# Metrics for each cluster
weights = []
precisions = []
recalls = []
f1s = []
cms = []

for i in range(n_clusters):
    y_true_binary = [1 if int(x) == i else 0 for x in y_true]
    y_pred_binary = [1 if int(x) == i else 0 for x in y_pred]

    weights.append(sum(y_true_binary))
    precisions.append(precision_score(y_true_binary, y_pred_binary))
    recalls.append(recall_score(y_true_binary, y_pred_binary))
    f1s.append(f1_score(y_true_binary, y_pred_binary))
    cms.append(confusion_matrix(y_true_binary, y_pred_binary))

# Mean metrics
mean_precision = sum(precisions) / n_clusters
mean_recall = sum(recalls) / n_clusters
mean_f1 = sum(f1s) / n_clusters

labels = data.obs['joint_leiden_clusters'].cat.categories.tolist()
big_cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=big_cm, display_labels=labels)
disp.plot()
plt.title("sample 1 multi-class confusion matrix")
plt.show()

correct = 0
for i in range(len(labels)):
    correct += big_cm[i][i]
accuracy = correct/len(data.obs)

for n, i in enumerate(cms):
    print(f"cluster {n}: {i}")
print(f"accuracy: {format(accuracy, '.4f')}\n\
precision: {tuple(format(p, '.4f') for p in precisions)}, {format(mean_precision, '.4f')}\n\
recall: {tuple(format(r, '.4f') for r in recalls)}, {format(mean_recall, '.4f')}\n\
f1_score: {tuple(format(f, '.4f') for f in f1s)}, {format(mean_f1, '.4f')}\n\

# Plotting ROC curves from prediction probabilities and calculating AUC scores
# Method modified from scikit-learn
# Combine pred probabilities from every fold's test set
test_pred = []
n = 0

for n in range(len(data.obs)):
    for j in range(10):
        for i in range(len(test_pred_probabilities[j][0])):
            if test_pred_probabilities[j][0][i] == n:
                test_pred.append(test_pred_probabilities[j][1][i])
                
# Onehot true labels
label_binarizer = LabelBinarizer().fit(data.obs.joint_leiden_clusters)
y_onehot_true = label_binarizer.transform(data.obs.joint_leiden_clusters)

colors = data.uns['joint_leiden_clusters_colors']
n_classes = len(labels)
y_score = np.array(test_pred)

# Calculate ROC curves and AUC scores
fig, ax = plt.subplots(figsize=(6, 6))
for class_id, color in zip(range(n_classes), colors):
    # Calculate AUC score with 4 decimal places
    auc_score = roc_auc_score(y_onehot_true[:, class_id], y_score[:, class_id])
    auc_score = "{:.4f}".format(auc_score)
    RocCurveDisplay.from_predictions(
        y_onehot_true[:, class_id],
        y_score[:, class_id],
        label=f"ROC curve for {class_id} (AUC: {auc_score})",
        color=color,
        ax=ax,
        plot_chance_level=(class_id == n_classes-1),
    )

# Calculate macro-average (mean) ROC and AUC
# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_true[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"mean ROC curve (AUC = {roc_auc['macro']:.4f})",
    color="navy",
    linestyle=":",
    linewidth=2,
)

# Plotting
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("sample 1 OvR multi-class ROC curves")
plt.legend()
plt.show()
