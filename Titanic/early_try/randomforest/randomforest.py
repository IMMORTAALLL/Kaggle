from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import torch

df = pd.read_csv(r"/workspace/kaggle/Titanic/data/raw/train.csv")
df1 = pd.read_csv(r"/workspace/kaggle/Titanic/data/raw/test.csv")
df.drop(columns=['Name','Ticket','PassengerId','Cabin'],inplace=True)
test_id = df1['PassengerId'].values
df1.drop(columns=['Name','Ticket','Cabin'],inplace=True)
df['Age'] = df['Age'].fillna(value=df['Age'].mean())
df1['Age'] = df1['Age'].fillna(value=df1['Age'].mean())
df['Sex'] = df['Sex'].map({"male":0,"female":1})
df1['Sex'] = df1['Sex'].map({"male":0,"female":1})
df['Embarked'] = df['Embarked'].fillna(value=df['Embarked'].mode()[0])
df1['Embarked'] = df1['Embarked'].fillna(value=df1['Embarked'].mode()[0])
df1['Fare'] = df1['Fare'].fillna(value=df1['Fare'].mode()[0])
embarked_encoded = pd.get_dummies(df["Embarked"], prefix="Embarked", drop_first=False,dtype=int)
df = pd.concat([df, embarked_encoded], axis=1)
embarked_encoded = pd.get_dummies(df1["Embarked"], prefix="Embarked", drop_first=False,dtype=int)
df1 = pd.concat([df1, embarked_encoded], axis=1)

df_X = df.drop(columns=['Survived','Embarked']).values
df_y = df['Survived'].values

df1_X = df1.drop(columns=['Embarked','PassengerId']).values


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=42, shuffle=True)
rf_clf = RandomForestClassifier(n_estimators=10000, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
print(f"准确率：{accuracy_score(y_test, y_pred):.2f}")

sub_pred = rf_clf.predict(df1_X)
print(sub_pred)
sub_df = pd.DataFrame({'PassengerId':df1['PassengerId'].values,'Survived':sub_pred})
sub_df.to_csv(r"submmit1.csv",index=False)