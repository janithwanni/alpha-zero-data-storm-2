from imblearn.over_sampling import SMOTENC
import pandas as pd

def oversample(dataframe: pd.DataFrame, cat_feats: List[int]):
    X = dataframe.drop("Reservation_Status",axis="columns")
    y = dataframe.loc[:,"Reservation_Status"]
    smote_enc = SMOTENC(categorical_features=cat_feats,random_state=42)
    X_res, y_res = smote_enc.fit_resample(X,y)
    out_df = X_res.copy(deep=True)
    out_df["Reservation_Status"] = y_res
    return out_df

train_df = pd.read_csv("../../data/processed/train_preproc.csv")
train_df = train_df.loc[:,["Reservation_Status","N_Minors","Total_PAX","Income_number","Cost","Cost_Income","Lag","Duration"]]

categorical_columns = ["Gender","Ethnicity","Educational_Level","Income","Country_region","Hotel_Type",
                       "Meal_Type","Visted_Previously","Previous_Cancellations","Deposit_type","Booking_channel",
                       "Required_Car_Parking","Use_Promotion","Visit_Cancel"]

cat_feats = [train_df.drop("Reservation_Status",axis="columns").columns.get_loc(col) for col in categorical_columns]
os_df = oversample(train_df,cat_feats)
os_df.to_csv("../../data/processed/oversampled/train_oversample.csv",index=False)