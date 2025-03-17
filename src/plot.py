import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

# Load data
test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")

# Define feature names
embeddings = [f"d_{i+1}" for i in range(320)]

# Split into features (X) and labels (y)
X_train = train_df[embeddings]
y_train = train_df["syndrome_id"]
X_test = test_df[embeddings]
y_test = test_df["syndrome_id"]

# Compute syndrome counts
syndrome_counts = y_test.value_counts().to_dict()

# One-hot encode labels (for ROC AUC)
encoded_y_test = pd.get_dummies(y_test)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=12, metric="cosine")
knn.fit(X_train, y_train)

# Get probability scores for each class
y_score = knn.predict_proba(X_test)

# Streamlit UI
st.title("Metrics of the KNN Model")

auc_values = []
roc_curves = {}

# Create figure
fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

for i, label in enumerate(encoded_y_test.columns):
    fpr, tpr, _ = roc_curve(encoded_y_test.iloc[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    instance_count = syndrome_counts.get(label, 0)  # Get instance count
    auc_values.append((roc_auc, label, instance_count, fpr, tpr))

# Sort by AUC descending
auc_values.sort(reverse=True, key=lambda x: x[0])

# Generate colors
palette = sns.color_palette("tab10", n_colors=len(auc_values))
color_map = {label: color for (_, label, _, _, _), color in zip(auc_values, palette)}

# Plot sorted ROC curves
fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
for (roc_auc, label, instance_count, fpr, tpr) in auc_values:
    ax.plot(fpr, tpr, label=f"Syndrome {label} (AUC = {roc_auc:.2f}, n={instance_count})", color=color_map[label])

# Plot reference line
ax.plot([0, 1], [0, 1], "k--", lw=2)

# Set labels and title
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC AUC Curve for Each Syndrome")
ax.legend(loc="lower right")

# Display plot in Streamlit
st.pyplot(fig)

# Create bar plot for AUC values with syndrome counts
df_auc = pd.DataFrame(auc_values, columns=["AUC", "Syndrome", "Count", "FPR", "TPR"])

# Seaborn barplot
fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
sns.barplot(
    data=df_auc,
    x="AUC",
    y="Syndrome",
    hue="Syndrome",
    palette=color_map,
    dodge=False,
    ax=ax
)

# Show counts inside bars
for i, (auc_value, label, count, _, _) in enumerate(auc_values):
    ax.text(auc_value + 0.01, i, f"n={count}", va="center", fontsize=10)

ax.set_xlabel("AUC Score")
ax.set_ylabel("Syndrome")
ax.set_title("AUC Scores Sorted with Instance Counts")
ax.legend_.remove()  # Remove duplicate legend

# Display bar plot in Streamlit
st.pyplot(fig)
