from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

id = (pd.read_csv(r"/workspace/kaggle/Titanic/data/raw/test.csv"))['PassengerId'].values

features = (pd.read_csv(r"/workspace/kaggle/Titanic/data/processed/train_processed.csv")).drop(columns=["Survived"]).values
labels = (pd.read_csv(r"/workspace/kaggle/Titanic/data/processed/train_processed.csv"))["Survived"].values
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.1, random_state=42)

if __name__ == "__main__":
    model = RandomForestClassifier(n_estimators=10000,random_state=42,criterion='gini',max_depth=24)
    model.fit(train_features, train_labels)
    outputs = model.predict(test_features)
    accuracy = accuracy_score(outputs, test_labels)
    print(f"准确率为:{accuracy}")

    predicts = model.predict(pd.read_csv(r"/workspace/kaggle/Titanic/data/processed/test_processed.csv").drop(columns=['Survived']).values)
    pd.DataFrame({"PassengerId":id, "Survived":predicts.astype(int)}).to_csv(r"/workspace/kaggle/Titanic/src/predict/ran_submit.csv", index=False)