import os
import sys
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    dataset = sys.argv[3] if len(sys.argv) > 3 else "train_pca.csv"

    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dataset)

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("addiction_binary", axis=1),
        data["addiction_binary"],
        random_state=42,
        test_size=0.2,
        stratify=data["addiction_binary"]
    )

    input_example = X_train.iloc[:5]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        print("Training MLflow Project selesai.")
        print("Accuracy:", accuracy)