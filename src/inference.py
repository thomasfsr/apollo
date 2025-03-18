from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import mlflow
import numpy as np

train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop(columns=["syndrome_id"], axis=1)
y_train = train_df["syndrome_id"]
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=["syndrome_id"],axis=1)
y_test = test_df["syndrome_id"]

def inference():
    model = KNN(13, metric='cosine')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results_df = pd.DataFrame({"y_test": y_test.values.flatten(), "y_pred": y_pred})
    results_df.to_csv("data/test_predictions.csv", index=False)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"F1 Score: {f1}")

if __name__ == "__main__":
    inference()