import pandas as pd
import re


train_df = pd.read_csv(r"kaggle/Titanic/data/raw/train.csv")
test_df = pd.read_csv(r"kaggle/Titanic/data/raw/test.csv")
all_df = pd.concat([train_df, test_df], ignore_index=True)


all_df["Embarked"] = all_df["Embarked"].fillna(value=all_df["Embarked"].mode()[0])
missing_fare = all_df[pd.isnull(all_df['Fare'])]['Pclass'].values[0]
all_df['Fare'] = all_df['Fare'].fillna(value=all_df[all_df['Pclass'] == missing_fare]['Fare'].median())

age_medians = all_df.groupby(['Pclass','Sex'])['Age'].median()

def fill_age(row):
    if pd.isnull(row['Age']):
        return age_medians.loc[(row['Pclass'], row['Sex'])]
    else:
        return row['Age']
    
all_df['Age'] = all_df.apply(fill_age, axis=1)
all_df['Cabin_First'] = all_df['Cabin'].apply(lambda x: "U" if pd.isnull(x) else x[0])
all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1
all_df['IsAlone'] = all_df['FamilySize'].apply(lambda x: 1 if x == 1 else 0)

def extract_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    else :
        return ""

all_df['Title'] = all_df['Name'].apply(extract_title)
title_counts = all_df['Title'].value_counts()
rare_title = title_counts[title_counts < 10].index
all_df['Title'] = all_df['Title'].replace({
    'Mlle': 'Miss',
    'Ms': 'Miss',
    'Mme': 'Mrs' 
})
all_df['Title'] = all_df['Title'].apply(lambda x: "Rare" if x in rare_title else x)

all_df['FareBin'] = pd.qcut(all_df['Fare'], 4, labels=['f0', 'f1', 'f2', 'f3'])
all_df['AgeBin'] = pd.cut(all_df['Age'], bins=[0,12,18,50,100], labels=['c','t','a','s'])

all_df['Sex'] = all_df['Sex'].map({"male":0,"female":1})

all_df.drop(columns=['Name', 'SibSp', 'Cabin', 'Ticket', 'Parch','Age','Fare','PassengerId'], inplace=True)



all_df = pd.get_dummies(all_df,dtype=int,drop_first=True)

processed_train_df = all_df[~all_df['Survived'].isnull()]
processed_test_df = all_df[all_df['Survived'].isnull()]

if __name__ == "__main__":
    print(train_df.info(), test_df.info())
    print(train_df)
    print(all_df.isnull().sum())
    print(age_medians)
    print(rare_title) 
    print(title_counts)
    print(all_df.info())
    all_df.to_csv(r"kaggle/Titanic/data/processed/all_processed.csv",index=False)
    processed_train_df.to_csv(r"kaggle/Titanic/data/processed/train_processed.csv",index=False)
    processed_test_df.to_csv(r"kaggle/Titanic/data/processed/test_processed.csv",index=False)
    