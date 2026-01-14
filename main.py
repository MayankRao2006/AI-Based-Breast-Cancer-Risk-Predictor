import pandas as pd
import joblib
import os

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

MODEL_FILE = "model.pkl"
PIPELINE = "pipeline.pkl"

def build_pipeline(num_attrs):
    num_pipeline = Pipeline([
        ('scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attrs),
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    df = pd.read_csv("data.csv")
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["Class"]):
        train_set = df.iloc[train_index]
        test_set = df.iloc[test_index]

    test_set.to_csv("Testing_data.csv", index=False)

    features = train_set.drop("Class", axis=1)
    labels = train_set["Class"].copy()

    num_attrs = features.columns

    pipeline = build_pipeline(num_attrs)
    prepared_data = pipeline.fit_transform(features)

    model = RandomForestClassifier(n_estimators=300,
    max_depth=6,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42)
    model.fit(prepared_data, labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE)

    print("Model training complete. Model and pipeline saved.")

if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE)

    test_df = pd.read_csv("Testing_data.csv")
    test_features = test_df.drop("Class", axis=1)
    test_labels = test_df["Class"].copy()

    prepared_test = pipeline.transform(test_features)

    predictions = model.predict(prepared_test)

    test_df["Predicted_Class"] = predictions
    test_df.to_csv("Predictions.csv", index=False)

    print("Inference complete. Predictions saved to Predictions.csv.")