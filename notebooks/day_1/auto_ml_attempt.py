from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

# train_df = pd.read_csv("../../data/processed/train_preproc.csv")
train_data = TabularDataset("../../data/processed/train_preproc.csv")
train_data = train_data.drop(["Age","Room_Rate","Discount_Rate"],axis="columns")
save_path = "models"
predictor = TabularPredictor(label="Reservation_Status",path=save_path,eval_metric="f1_macro").fit(train_data,time_limit=4800,
                                                                                             presets="best_quality_with_high_quality_refit")

valid_data = TabularDataset("../../data/processed/valid_preproc.csv")
y_test = valid_data.loc[:,"Reservation_Status"]
valid_data = valid_data.drop(["Reservation_Status","Age","Room_Rate","Discount_Rate"],axis="columns")

y_pred = predictor.predict(valid_data)
perf = predictor.evaluate_predictions(y_true=y_test, y_pred=y_pred, auxiliary_metrics=True)
print(perf)