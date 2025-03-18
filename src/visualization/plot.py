import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import roc_curve, auc

test_df = pd.read_csv("data/test.csv")
train_df = pd.read_csv("data/train.csv")
infered_df = pd.read_csv("data/test_predictions.csv")
df = pd.read_csv("data/df.csv")
tsne_df = pd.read_csv('data/tsne_df.csv')
results = pd.read_csv('data/results.csv')

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
st.markdown("""
To optimize the KNN Classifier we can try out the combination of hyperparameters
such as:  
            - Metric of distance: Euclidean and Cosine.  
            - Number of neighbors: 1 to 15.  
The distance change how the magnitude and dimensionality of the data impacts the model.
Euclidean distance are very sensitive to scale and high-dimensionality of the data.
Cosine distance is more robust since it focus on the angle of the vectors.
Moreover, there is another import hyperparameter for KNN model.  
The number of neighbors is inversionally proportional to the overfitting
of the model.  
The smaller K number, more prone to high variance (overfit), leading the
model to capture every noise in the data, memorizing the training data.  
  
Let us see the result of the different combinations.  
""")

st.dataframe(results)

st.markdown("""
The best model was with k=13 and cosine distance, achieving the following performance metrics:  
Overall:  
F1-score: 0.786  
Top 1-k accuracy: 0.805  
ROC AUC: 0.966  
As expected, the Cosine Distance and higher k performed better. The model was trained with 80 percent of the data, stratified by the syndrome label to achieve higher accuracy for the smaller classes. The remaining data will be used for final testing, validating the model and helping us visualize the ROC curve for each syndrome.  
""")
st.image('image/mlflow.png', width=1500)
encoded_y_train = pd.get_dummies(y_train)
encoded_y_test = pd.get_dummies(y_test)

syndrome_counts = y_test.value_counts().to_dict()

knn = KNN(n_neighbors=13, metric="cosine")
knn.fit(X_train, y_train)

# Get probability scores for each class
y_score = knn.predict_proba(X_test)

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
st.subheader("Conclusion")
st.markdown(
    '''
    To conclude, since the embedding probably was extracted
    using CNN pre-trained model such as RESNET, the semantic
    information in the vectors helped to improve the model using
    cosine distance to achieve acceptable results.

    What can be further exploring is to try out 
    a pre-trained model with a output layer for classification task.
      
    Moreover, there are other types of classifiers that are also
    robust against high-dimensionality such as LightGBM and XGBoost.  
    '''
)