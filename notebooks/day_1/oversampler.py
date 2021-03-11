from imblearn.over_sampling import SMOTENC
import pandas as pd
from typing import List
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

def oversample(dataframe: pd.DataFrame, cat_feats: List[int]):
    X = dataframe.drop("Reservation_Status",axis="columns")
    y = dataframe.loc[:,"Reservation_Status"]
    smote_enc = SMOTENC(categorical_features=cat_feats,random_state=0)
    X_res, y_res = smote_enc.fit_resample(X,y)
    return X_res,y_res


train_df = pd.read_csv("../../data/processed/train_preproc.csv")
categorical_columns = ["Gender","Ethnicity","Educational_Level","Income","Country_region","Hotel_Type",
                       "Meal_Type","Visited_Previously","Previous_Ca ncellations","Deposit_type","Booking_channel",]
# numerical_columns = ['age', 'weigth', ']
# column_trans = make_column_transformer(
#     (categorical_columns, OneHotEncoder(handle_unknown='ignore'),
#     (numerical_columns, RobustScaler())
# column_trans.fit_transform(df)
X_r ,y_r = oversample(train_df,[0,2,3,4,5,6,10,11,12,13,14,15])