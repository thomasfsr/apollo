from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

embeddings = [f"d_{i+1}" for i in range(320)]

df = pd.read_csv("data/df.csv")

scaler = StandardScaler()
scaled_df = df.copy()
scaled_df[embeddings] = scaler.fit_transform(df[embeddings])

emb_tsne = {'c_1':[], 'c_2':[], 'syndrome_id':[]}
tsne = TSNE(n_components=2, random_state=42, perplexity=100)
embeddings_tsne = tsne.fit_transform(scaled_df[embeddings])
emb_tsne['c_1'] = embeddings_tsne[:,0]
emb_tsne['c_2'] = embeddings_tsne[:,1]
emb_tsne['syndrome_id'] = df['syndrome_id']
df_tsne = pd.DataFrame(emb_tsne)

pca = PCA(n_components=20)
pca_embeddings = pca.fit_transform(scaled_df[embeddings])
pca_df = pd.DataFrame(pca_embeddings)
pca_df['syndrome_id'] = df['syndrome_id']

if __name__ == "__main__":
    df_tsne.to_csv('data/tsne_df.csv')
    pca_df.to_csv('data/pca_df.csv')
