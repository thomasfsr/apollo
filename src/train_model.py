import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import (f1_score,
                             top_k_accuracy_score,
                             roc_auc_score,
                             make_scorer)
import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

class Model:
    def __init__(self, input_path:str, output_path:str):
        self.input_path = input_path
        self.output_path = output_path

    def load(self)->pd.DataFrame:
        return pd.read_csv(self.input_path)

    def preprocess(self, df:pd.DataFrame)->pd.DataFrame:
        embeddings = [f'd_{i+1}' for i in range(320)]
        scaler = StandardScaler()
        df[embeddings] = scaler.fit_transform(df[embeddings])
        return df

    def split(self, df:pd.DataFrame):
        X = df.drop(columns=['syndrome_id', 'subject_id', 'image_id'])
        y = df['syndrome_id']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def train(self, X:pd.DataFrame, y:pd.DataFrame)->pd.DataFrame:
        results = {'distance':[], 'k':[], 'f1':[], 'topk':[], 'auc':[]}
        distances = ['euclidean', 'cosine']
        n_neighbors = np.arange(1, 16)
        top1_k_score = make_scorer(top_k_accuracy_score, response_method='predict_proba', k=1)

        for distance in distances:
            for k in n_neighbors:
                knn = KNeighborsClassifier(n_neighbors=k, metric=distance)
                f1 = cross_val_score(knn, X, y, cv=10, scoring ='f1_macro')
                topk = cross_val_score(knn, X, y, cv=10, scoring = top1_k_score)
                auc = cross_val_score(knn, X, y, cv=10, scoring ='roc_auc_ovr_weighted')
                print(f'Distance: {distance}, K: {k}, F1: {f1.mean()}, Top-k: {topk.mean()}, AUC: {auc.mean()}')
                results['distance'].append(distance)
                results['k'].append(k)
                results['f1'].append(f1.mean())
                results['topk'].append(topk.mean())
                results['auc'].append(auc.mean())
        
        results = pd.DataFrame(results)
        return results

    def ml_train(self, X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
        results = {'distance': [], 'k': [], 'f1': [], 'topk': [], 'auc': []}
        distances = ['euclidean', 'cosine']
        n_neighbors = np.arange(1, 16)
        top1_k_score = make_scorer(top_k_accuracy_score, response_method='predict_proba', k=1)

        mlflow.set_experiment("KNN_Hyperparameter_Tuning")

        for distance in distances:
            for k in n_neighbors:
                with mlflow.start_run(): 
                    knn = KNeighborsClassifier(n_neighbors=k, metric=distance)

                    f1 = cross_val_score(knn, X, y, cv=10, scoring='f1_macro')
                    topk = cross_val_score(knn, X, y, cv=10, scoring=top1_k_score)
                    auc = cross_val_score(knn, X, y, cv=10, scoring='roc_auc_ovr_weighted')

                    f1_mean, topk_mean, auc_mean = f1.mean(), topk.mean(), auc.mean()

                    print(f"Distance: {distance}, K: {k}, F1: {f1_mean:.4f}, Top-k: {topk_mean:.4f}, AUC: {auc_mean:.4f}")

                    mlflow.log_param("distance", distance)
                    mlflow.log_param("k", k)
                    mlflow.log_metric("f1", f1_mean)
                    mlflow.log_metric("topk", topk_mean)
                    mlflow.log_metric("auc", auc_mean)

                    mlflow.sklearn.log_model(knn, f"knn_distance_{distance}_k_{k}")

                results['distance'].append(distance)
                results['k'].append(k)
                results['f1'].append(f1_mean)
                results['topk'].append(topk_mean)
                results['auc'].append(auc_mean)

        results_df = pd.DataFrame(results)
        return results_df

    def to_csv(self, result:pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            result.to_csv(self.output_path, index=False)
            print(f"Data saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")
        pass

if __name__ == '__main__':
    model = Model('data/emb.csv', 'data/results.csv')
    df = model.load()
    df = model.preprocess(df)
    X_train, X_test, y_train, y_test = model.split(df)
    test = pd.concat([X_test, y_test], axis=1)
    train = pd.concat([X_train, y_train], axis=1)
    test.to_csv('data/test.csv', index=False)
    train.to_csv('data/train.csv', index=False)
    results = model.ml_train(X_train, y_train)
    model.to_csv(results)