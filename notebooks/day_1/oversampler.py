from imblearn.over_sampling import SMOTENC
import pandas as pd
from typing import List

train_df = pd.read_csv("../../data/processed/train_preproc.csv")

def oversample(dataframe: pd.DataFrame, cat_feats: List[int]):
    X = dataframe.drop("Reservation_Status",axis="columns")
    y = dataframe.loc[:,"Reservation_Status"]
    smote_enc = SMOTENC(categorical_features=cat_feats,random_state=0)
    X_res, y_res = smote_enc.fit_resample(X,y)
    return X_res,y_res
