import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import shapiro
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.
from scipy import stats

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")
infered_df = pd.read_csv("data/test_predictions.csv")
df = pd.read_csv("data/df.csv")
tsne_df = pd.read_csv('data/tsne_df.csv')
pca_df = pd.read_csv('data/pca_df')

embeddings = [f"d_{i+1}" for i in range(320)]

X_train = train_df[embeddings]
y_train = train_df["syndrome_id"]
X_test = test_df[embeddings]
y_test = test_df["syndrome_id"]


st.title("Apollo ML Project")
st.subheader("Exploratory Data Analysis")
st.write('Lets start by going over the distribution of the syndromes and do some validation check (e.g. null values).')

if df.isnull().sum().sum() == 0:
    st.metric(label="Missing Values", value="None ✅")

st.subheader("Distribution of Syndrome")
st.write('The distribution of the syndromes is as follows:')
grouped_df = df[['syndrome_id', 'image_id']].groupby('syndrome_id').count().sort_values('image_id', ascending=False).reset_index()
grouped_df['syndrome_id'] = pd.Categorical(grouped_df['syndrome_id'], categories=grouped_df['syndrome_id'], ordered=True)
st.subheader("Count of Images per Syndrome ID")
st.bar_chart(grouped_df.set_index('syndrome_id'), x_label='syndrome_id', y_label='Count of images')
st.markdown("As we can see the distribution of syndromes in the dataset is not heavily imbalanced.")

st.subheader("Checking the normality of the embeddings using Shapiro Test")
shapiro_test = {'emb':[], 'stat':[], 'p':[], 'is_normal':[]}
for i in embeddings:
    shapiro_test['emb'].append(i)
    stat, p = shapiro(df[i])
    shapiro_test['stat'].append(stat)
    shapiro_test['p'].append(p)
    if p < 0.05:
        shapiro_test['is_normal'].append(False)
    else:
        shapiro_test['is_normal'].append(True)
df_shapiro = pd.DataFrame(shapiro_test)

proportion_normal = df_shapiro['is_normal'].value_counts().sort_values()
c1, c2 = st.columns(2)
with c1:
    plt.figure(figsize=[10,5])
    st.bar_chart(proportion_normal)

prop = df_shapiro['is_normal'].value_counts()/320
prop = prop.rename_axis("Normality").rename("Percentage of Gaussian Data")
with c2:
    st.dataframe(prop, use_container_width=False)

st.subheader("Exploring Dimensional Reduction with T-SNE")
st.markdown("t-SNE is useful for capturing relationships among high-dimensional data points while reducing dimensionality, preserving their proximity.")

st.subheader('Scalling Data')
st.markdown("First, the embedding data was scaled to a standard distribution with a mean of zero and a standard deviation of one.")
scaled_df = df.copy()
scaler = StandardScaler()
scaled_df[embeddings] = scaler.fit_transform(df[embeddings])

st.dataframe(scaled_df[embeddings].describe())

st.markdown("""After the reduction of dimensionality to 2-D it is possible to plot the datapoints
and visualize possible correlation of the embeddings of the same syndrome.""")

fig, ax = plt.subplots(figsize=(10, 10))
sns.scatterplot(data=tsne_df, x='c_1', y='c_2', 
                hue='syndrome_id', 
                palette='tab20', 
                ax=ax)
ax.set_xlabel("Component 1")
ax.set_ylabel("Component 2")
ax.set_title('T-SNE on Embeddings')
st.pyplot(fig)

st.markdown("""The proximity of data points within the same syndrome suggests
             a correlation between the embedding and the syndrome category. 
            Despite the presence of outliers, this further confirms that the
             data distribution exhibits long-tailed characteristics.""")

st.subheader('K- Nearest Neighbors Classifier (KNN)')
st.markdown("""To the task of predicting the syndrome using the embeded image there
            is a model
""")

# Train KNN model
X, y = pca_df[:-1], pca_df['syndrome_id']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
knn = KNeighborsClassifier(n_neighbors=12, metric="cosine")
knn.fit(X_train, y_train)

# # Get probability scores for each class
# y_score = knn.predict_proba(X_test)


# auc_values = []
# roc_curves = {}

# # Create figure
# fig, ax = plt.subplots(figsize=(10, 7), dpi=200)

# for i, label in enumerate(encoded_y_test.columns):
#     fpr, tpr, _ = roc_curve(encoded_y_test.iloc[:, i], y_score[:, i])
#     roc_auc = auc(fpr, tpr)
#     instance_count = syndrome_counts.get(label, 0)  # Get instance count
#     auc_values.append((roc_auc, label, instance_count, fpr, tpr))

# # Sort by AUC descending
# auc_values.sort(reverse=True, key=lambda x: x[0])

# # Generate colors
# palette = sns.color_palette("tab10", n_colors=len(auc_values))
# color_map = {label: color for (_, label, _, _, _), color in zip(auc_values, palette)}

# # Plot sorted ROC curves
# fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
# for (roc_auc, label, instance_count, fpr, tpr) in auc_values:
#     ax.plot(fpr, tpr, label=f"Syndrome {label} (AUC = {roc_auc:.2f}, n={instance_count})", color=color_map[label])

# # Plot reference line
# ax.plot([0, 1], [0, 1], "k--", lw=2)

# # Set labels and title
# ax.set_xlabel("False Positive Rate")
# ax.set_ylabel("True Positive Rate")
# ax.set_title("ROC AUC Curve for Each Syndrome")
# ax.legend(loc="lower right")

# # Display plot in Streamlit
# st.pyplot(fig)

# # Create bar plot for AUC values with syndrome counts
# df_auc = pd.DataFrame(auc_values, columns=["AUC", "Syndrome", "Count", "FPR", "TPR"])

# # Seaborn barplot
# fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
# sns.barplot(
#     data=df_auc,
#     x="AUC",
#     y="Syndrome",
#     hue="Syndrome",
#     palette=color_map,
#     dodge=False,
#     ax=ax
# )

# # Show counts inside bars
# for i, (auc_value, label, count, _, _) in enumerate(auc_values):
#     ax.text(auc_value + 0.01, i, f"n={count}", va="center", fontsize=10)

# ax.set_xlabel("AUC Score")
# ax.set_ylabel("Syndrome")
# ax.set_title("AUC Scores Sorted with Instance Counts")
# ax.legend_.remove()  # Remove duplicate legend

# # Display bar plot in Streamlit
# st.pyplot(fig)
