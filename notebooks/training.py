from pathlib import Path
import json
import openmeteo_requests
from dotenv import load_dotenv
import datetime
import hopsworks
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from xgboost import XGBRegressor
from xgboost import plot_importance

#setUp: 
HOPSWORKS_HOST="c.app.hopsworks.ai"
#May need https://
HOPSWORKS_PROJECT="AirQualityKTHLab1"
city = "stockholm"
pollen_types = ["alder_pollen","birch_pollen","grass_pollen" ,"mugwort_pollen" ,"olive_pollen" ,"ragweed_pollen"]
#Stockholm: 59.334591, 18.063240
#new: 59.338295,18.055603
latitude =59.338295
longitude = 18.055603

today = datetime.date.today()
print("today: " , today)
yesturday =  today - datetime.timedelta(days=1)
print("yesturday: " , yesturday)
root_dir = Path.cwd()
print(root_dir)
load_dotenv()
project = hopsworks.login(host=HOPSWORKS_HOST, project=HOPSWORKS_PROJECT)

fs = project.get_feature_store() 
mr = project.get_model_registry()


#FUNCTION FROM MY LAB1 IN THIS COURSE:
def forecast_recursive(model, row_today, rows_to_predict, feature_cols, max_horizon, use_lags, pollen_type):
    predictions = []
    
    if use_lags: #establish a starting point (today!): 
        lag_1 = row_today[pollen_type]
        lag_2 = row_today[f"{pollen_type}_lag1"]
        lag_3 = row_today[f"{pollen_type}_lag2"]
        
    for h in range(max_horizon): #from tomorrow onwards: 
        row_to_predict = rows_to_predict.iloc[h].copy()
        
        if use_lags: #overwrite possible real values (which is illegal to use here) with predicted values!
            row_to_predict[f"{pollen_type}_lag1"] = lag_1
            row_to_predict[f"{pollen_type}_lag2"] = lag_2
            row_to_predict[f"{pollen_type}_lag3"] = lag_3
            
        X_row = row_to_predict[feature_cols].values.reshape(1, -1)
        y_pred = float(model.predict(X_row)[0])

        predictions.append(y_pred)
        if use_lags:
            #push all lags back because we move forward one day:
            lag_3 = lag_2 
            lag_2 = lag_1
            lag_1 = y_pred

        
    return predictions

# Compute MSE1,...,MSE7: (FUNCTION FROM MY LAB1 IN THIS COURSE:)
def hindcast_recursive(df, model, feature_cols, max_horizon=7, use_lags=False, pollen_type = "NOT_TYPE"):
    if pollen_type =="NOT_TYPE":
        print("NO TYPE GIVEN")
        return None
    
    target_col=pollen_type
    df = df.sort_values("date").reset_index(drop=True)
    
    records= []
    
    for start_index in range(len(df) - max_horizon): #we dont want to 
        row_today = df.iloc[start_index].copy()
        rows_to_predict = df.iloc[start_index+1:start_index+1+max_horizon].copy()
        
        y_predictions = forecast_recursive(model, row_today, rows_to_predict, feature_cols, max_horizon, use_lags, pollen_type)

        #unpack all:
        for i in range(max_horizon):
            y_pred = y_predictions[i]
            target_row = rows_to_predict.iloc[i]
            y_true = float(target_row[target_col])

            records.append({
                "forecast_start_date": row_today['date'],
                "target_date": target_row['date'],
                "horizon": i+1,
                "y_true": y_true,
                "y_pred": y_pred,
            })


    return pd.DataFrame(records)

#plotting function inspired by the lab 1: plot_air_quality_forecast in util.py:
def plot_hindcast(df_hindcast, pollen_type, file_path):
    date =pd.to_datetime(df_hindcast["date"]).dt.date
    fig, ax =plt.subplots(figsize=(12, 6))

    ax.set_title(f"1-day ahead hindcast for {pollen_type}")
    ax.set_ylabel(f"Pollen level - {pollen_type}")
    ax.set_xlabel("Date")

    ax.plot(
        date,
        df_hindcast[pollen_type],
        marker='^',
        markersize=4,
        label=f"Actual {pollen_type}",
        linewidth=2,
        color='black'
    )
    ax.plot(
        date,
        df_hindcast["predictions"],
        markersize=4,
        label=f"Predicted  {pollen_type}",
        marker='o',
        linewidth=2,
        color='blue'
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    plt.xticks(rotation=45)
    ax.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

def train_model(pollen_type, test_start):
    print("Training model for: ", pollen_type)

    #In this function, we will reuse some of the code from lab 1:
    pollen_fg = fs.get_feature_group(
        name='pollen',
        version=5,
    )
    wether_fg = fs.get_feature_group(
        name='weather',
        version=5,
    )
    pollen_query= pollen_fg.select(["date", pollen_type, f"{pollen_type}_lag1", f"{pollen_type}_lag2", f"{pollen_type}_lag3"])
    #df_pollen= df_pollen.dropna()
    
    merged_query = pollen_query.join(wether_fg.select_all(), on=["date"])

    fv_name = f"fv_{pollen_type}"

    feature_view = fs.get_or_create_feature_view(
        name = fv_name,
        description="weather and pollen data. Pollen level is target. Including lagged levels as features.",
        version=5,
        labels=[pollen_type],
        query=merged_query,
    )
    #Again, resuing some of the code from lab 1: 
    X_train, X_test, y_train, y_test = feature_view.train_test_split(test_start=test_start)

    training_mask = X_train.notna().all(axis=1) &y_train.notna().all(axis=1) #want to ensure that we only train and test on examples for which the data is complete!
    testing_mask  = X_test.notna().all(axis=1) &y_test.notna().all(axis=1)

    print("BEFORE MASK")
    print("X_train:")
    print(X_train)
    print("y_train:")
    print(y_train)

    #masking:
    X_train = X_train[training_mask]
    y_train = y_train[training_mask]
    X_test = X_test[testing_mask]
    y_test = y_test[testing_mask]

    print("AFTER MASK")
    print("X_train:")
    print(X_train)
    print("y_train:")
    print(y_train)
    X_train_features = X_train.drop(columns=['date'])
    X_test_features = X_test.drop(columns=['date'])

    #Training and evaluating the model:
    xgb_model = XGBRegressor()
    xgb_model.fit(X_train_features, y_train)
    y_predictions = xgb_model.predict(X_test_features)

    mse = mean_squared_error(y_test.iloc[:,0], y_predictions)       #Also taken from Lab 1!
    print(f"Model {pollen_type} MSE:", mse)
    
    r2 = r2_score(y_test.iloc[:,0], y_predictions)
    print(f"Model {pollen_type} R squared:", r2)


    #Eval for each time horizont (again, reusing much of the code from my lab 1):
    print(len(X_test) == len(y_test))
    df_test = X_test.copy()
    df_test[pollen_type] = y_test[pollen_type].values
    df_test["date"]  = pd.to_datetime(df_test["date"]) 
    df_test = df_test.sort_values("date").reset_index(drop=True)
    print(df_test[60:110])

    relevant_cols = list(X_train_features.columns)

    hc = hindcast_recursive(
        df_test,
        model=xgb_model,
        feature_cols=relevant_cols,
        max_horizon=7,
        use_lags=True,
        pollen_type=pollen_type
    )

    def mse_group(group):
        return mean_squared_error(group.y_true, group.y_pred)
        
    def mse_per_horizon(hc_df):
    
        return hc_df.groupby("horizon").apply(mse_group)
    
    mse_h = mse_per_horizon(hc)

    #Now we need to save the model
    res_dict  = { 
        "MSE": str(mse),
        "R squared": str(r2),
    }

    for h, mse in mse_h.items():
        res_dict[f"MSE_h{h}"] = str(mse)
    print(res_dict)

    model_dir = f"pollen_model_{pollen_type}"

    #creating the relevant dirs: (also taken from the lab 1)
    if not os.path.exists(model_dir): os.mkdir(model_dir)
    images_dir = model_dir + "/images"

    if not os.path.exists(images_dir): os.mkdir(images_dir)

    #create a hindcast df with predicted values, actual values and the date for these.: 
    df_hindcast = y_test.copy()
    df_hindcast["predictions"] = y_predictions
    df_hindcast["date"] = X_test["date"]
    df_hindcast = df_hindcast.sort_values("date").reset_index(drop=True)
    
    #Now just plot it:
    plot_hindcast(df_hindcast, pollen_type, images_dir + "/hindcast.png")

    #Feature importace: 
    plot_importance(xgb_model)
    plt.savefig(images_dir + "/features.png")
    plt.show()

    xgb_model.save_model(model_dir + "/model.json")

    model_name = f"model_{pollen_type}"

    pollen_model = mr.python.create_model(
        name = model_name,
        metrics= res_dict,
        feature_view=feature_view,
        description= f"Models {pollen_type}",
    )

    pollen_model.save(model_dir)

def main():
    start_date_test_data = "2025-01-01"
    # Convert string to datetime object
    test_start = datetime.datetime.strptime(start_date_test_data, "%Y-%m-%d")

    print("Training with test data staring at: " + str(test_start))

    for pollen_type in pollen_types:
        train_model(pollen_type, test_start)

    print("Training completed.")

if __name__ == "__main__":
    main()


