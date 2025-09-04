import xgboost as xgb
from randomforest import train_features, train_labels, test_features, test_labels, id
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

param_grid = {
    'max_depth':[3,5,8,10],
    'learning_rate':[0.001,0.01,0.05,0.1],
    'n_estimators':[100,500,1000],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(train_features, train_labels)
print(f"最佳参数{grid_search.best_params_}")
print(f"最佳准确率{grid_search.best_score_}")

best_model = grid_search.best_estimator_
pred = best_model.predict(test_features)
print(accuracy_score(pred, test_labels))

predicts = best_model.predict(pd.read_csv(r"/workspace/kaggle/Titanic/data/processed/test_processed.csv").drop(columns=['Survived']).values)
pd.DataFrame({"PassengerId":id, "Survived":predicts.astype(int)}).to_csv(r"/workspace/kaggle/Titanic/src/predict/xgb_submit.csv", index=False)