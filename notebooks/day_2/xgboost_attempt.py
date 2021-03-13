import xgboost as xgb
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix

train_df = pd.read_csv("../../data/processed/oversampled/train_feat_eng_oversample.csv")
valid_df = pd.read_csv("../../data/processed/valid_preproc.csv")
test_df = pd.read_csv("../../data/processed/test_preproc.csv")

train_X = pd.get_dummies(train_df.drop("Reservation_Status",axis="columns"))
train_Y = pd.get_dummies(train_df.loc[:,"Reservation_Status"])

valid_X = pd.get_dummies(valid_df.drop("Reservation_Status",axis="columns"))
valid_Y = pd.get_dummies(valid_df.loc[:,"Reservation_Status"])

xg_train = xgb.DMatrix(train_X,train_Y,enable_categorical=True)
xg_valid = xgb.DMatrix(valid_X,valid_Y,enable_categorical=True)

param = {"eta":0.1,"max_depth":6,"nthread":4,"num_class":3,"objective":"multi:softmax"}

watchlist = [(xg_train,'train'),(xg_valid,'test')]
num_round = 5
bst = xgb.train(param,xg_train,num_round,watchlist)
pred = bst.predict(xg_valid)

print(f1_score(y_true=valid_Y,y_pred=pred,average="macro"))
print(confusion_matrix(y_true=valid_Y,y_pred=pred))