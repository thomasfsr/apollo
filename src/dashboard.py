import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv('data/train.csv')
embeddings = [f'd_{i+1}' for i in range(320)]
X_train = train_df[embeddings]
y_train = train_df['syndrome_id']
X_test = test_df[embeddings]
y_test = test_df['syndrome_id']

encoded_y_train = pd.get_dummies(y_train)
encoded_y_test = pd.get_dummies(y_test)

knn = KNeighborsClassifier(n_neighbors= 12, metric='cosine')
knn.fit(X_train, y_train)

y_score = knn.predict_proba(X_test)

# Plot ROC AUC for each class
if __name__ == '__main__':

    plt.figure(figsize=(10, 7))

    for i, label in enumerate(encoded_y_test.columns):
        fpr, tpr, _ = roc_curve(encoded_y_test.iloc[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'Class {label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC AUC Curve for Each Class")
    plt.legend(loc="lower right")

    plt.show()



# def app():
#     st.title("Classification of Embeddings")