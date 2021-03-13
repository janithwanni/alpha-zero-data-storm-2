from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
from datetime import datetime

# train_df = pd.read_csv("../../data/processed/train_preproc.csv")
train_data = TabularDataset("../../data/processed/oversampled/train_valid_feat_eng_oversample.csv")
# train_data = train_data.drop(["Age","Room_Rate","Discount_Rate"],axis="columns")
save_path = "models_oversample_valid"
predictor = TabularPredictor(label="Reservation_Status",path=save_path,eval_metric="f1_macro").fit(train_data,
                                                                                                   time_limit=7200,
                                                                                                   presets="best_quality")

valid_data = TabularDataset("../../data/processed/valid_preproc.csv")
y_test = valid_data.loc[:,"Reservation_Status"]
valid_data = valid_data.drop(["Reservation_Status"],axis="columns")

y_pred = predictor.predict(valid_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print(perf)

test_data = TabularDataset("../../data/processed/test_preproc.csv")
test_preds = predictor.predict(test_data)

test_df = pd.read_csv("../../data/processed/test_preproc.csv")
test_df["Reservation_Status"] = test_preds
test_df = test_df.replace({"Reservation_Status":{"Check-In":1,"Canceled":2,"No-Show":3}})
test_df = test_df.loc[:,["Reservation-id","Reservation_Status"]]

test_df.to_csv("../../data/submissions/automl_"+str(datetime.now())+".csv",index=False)