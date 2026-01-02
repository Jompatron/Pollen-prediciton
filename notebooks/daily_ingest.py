from pathlib import Path
import openmeteo_requests
from dotenv import load_dotenv
import datetime
import hopsworks
import pandas as pd


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

def fetch_wether_forecasts(latitude, longitude):
    open_meteo = openmeteo_requests.Client()
    
    
    daily_variables = ["temperature_2m_mean", "precipitation_sum", "dew_point_2m_mean", "wind_speed_10m_mean", "sunshine_duration"]
    params ={
        "timezone": "Europe/Berlin",
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_variables,
        "forecast_days": 9
    }

    url = "https://api.open-meteo.com/v1/forecast"


    response = open_meteo.weather_api(url, params=params)[0]

    daily= response.Daily()

    #Unpacking the variables similarly as to how we did in Lab1 (from https://github.com/featurestorebook/mlfs-book/)
    daily_time = pd.date_range(start =pd.to_datetime(daily.Time(), unit = "s", utc=True),
        end = pd.to_datetime(daily.TimeEnd(),unit = "s", utc=True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )
    daily_time = daily_time.tz_convert("Europe/Stockholm")

    data = {
        "date": daily_time,
        "temperature_2m_mean": daily.Variables(0).ValuesAsNumpy(),
        "precipitation_sum": daily.Variables(1).ValuesAsNumpy(),
        "dew_point_2m_mean": daily.Variables(2).ValuesAsNumpy(),
        "wind_speed_10m_mean": daily.Variables(3).ValuesAsNumpy(),
        "sunshine_duration": daily.Variables(4).ValuesAsNumpy(),
    }

    df = pd.DataFrame(data)
    df["date"] =df["date"].dt.tz_localize(None).dt.normalize()
    return df

def fetch_yesturday_pollen(latitude, longitude):
    #https://air-quality-api.open-meteo.com/v1/air-quality?latitude=52.52&longitude=13.41&hourly=alder_pollen,birch_pollen,grass_pollen,mugwort_pollen,olive_pollen,ragweed_pollen&timezone=Europe%2FBerlin&start_date=2022-12-25&end_date=2026-01-06

    open_meteo = openmeteo_requests.Client()
    
    params ={
        "timezone": "Europe/Berlin",
        "start_date": str(yesturday),
        "end_date": str(yesturday),
        "latitude": latitude,
        "longitude": longitude,
        "hourly": pollen_types,
    }

    url = "https://air-quality-api.open-meteo.com/v1/air-quality"


    response = open_meteo.weather_api(url, params=params)[0]

    hourly= response.Hourly()
    #n_steps = len(hourly.Variables(0).ValuesAsNumpy())

    #Unpacking the variables similarly as to how we did in Lab1 (from https://github.com/featurestorebook/mlfs-book/)
    time_hourly = pd.date_range(start =pd.to_datetime(hourly.Time(), unit ="s", utc=True),
        end = pd.to_datetime(hourly.TimeEnd(),unit = "s", utc=True),
        #periods = n_steps,
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )
    time_hourly = time_hourly.tz_convert("Europe/Stockholm")

    data = {
        "date_time": time_hourly,
    }
    #unpacking all the data:
    for i, var in enumerate(pollen_types):
        data[var] = hourly.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data)
    
    df["date"] = df["date_time"].dt.tz_localize(None).dt.normalize()

    df_daily_max  = df.groupby("date")[pollen_types].max().reset_index()
    df_daily_max = df_daily_max.sort_values("date").reset_index(drop=True)
    wanted_columns = ["date"] + pollen_types

    return df_daily_max[wanted_columns]


def ingestWether():
    #wether: 
    df_wether = fetch_wether_forecasts(latitude, longitude)
    df_wether = df_wether.sort_values("date").reset_index(drop=True)

    df_wether['date'] = pd.to_datetime(df_wether['date']).dt.date
    df_wether['date'] = pd.to_datetime(df_wether['date'])
    print(df_wether)

    fs = project.get_feature_store() 
    #feature engineering: 
    wether_fg = fs.get_feature_group(
        name='weather',
        version=5,
    )

    #a bit form my lab 1: 
    history = wether_fg.read()
    history = history[history["date"].dt.date < today]

    context = history[history["date"].dt.date >(today-datetime.timedelta(days=16))].copy()

    sorted_context = context.sort_values("date").reset_index(drop=True)
    
    sorted_context['date'] = pd.to_datetime(sorted_context['date']).dt.date
    sorted_context['date'] = pd.to_datetime(sorted_context['date'])

    df_combined = pd.concat([sorted_context, df_wether], ignore_index=True).sort_values("date").reset_index(drop=True)

    #Feature engineering-------------------------
    df_combined["doy"] = df_combined["date"].dt.dayofyear
    #temp:
    df_combined["temp_7d_mean"] = df_combined["temperature_2m_mean"].rolling(window=7,min_periods=1).mean()
    df_combined["temp_14d_mean"] = df_combined["temperature_2m_mean"].rolling(window=14, min_periods=1).mean()
    #percipitaion:
    df_combined["precip_3d_sum"] = df_combined["precipitation_sum"].rolling(window=3, min_periods=1).sum()
    df_combined["precip_7d_sum"] = df_combined["precipitation_sum"].rolling(window=7, min_periods=1).sum()
    df_combined["dew_point_3d_mean"] = df_combined["dew_point_2m_mean"].rolling(window=3, min_periods=1).mean()
    #-------------------------------------------------------------

    
    df_to_insert = df_combined[df_combined["date"].dt.date >= today].copy()
    print("Feature enginnered new insert: ")
    print(df_to_insert) 

    wether_fg.insert(df_to_insert, wait=True)


def ingest_pollen():
    #pollen. We only want true values here. So lets only ingest yesturdays values:
    df_pollen = fetch_yesturday_pollen(latitude, longitude)
    df_pollen = df_pollen.sort_values("date").reset_index(drop=True)
    df_pollen['date'] = pd.to_datetime(df_pollen['date']).dt.date
    df_pollen['date'] = pd.to_datetime(df_pollen['date'])

    print(df_pollen)


    fs = project.get_feature_store() 
    #feature engineering: 
    pollen_fg = fs.get_feature_group(
        name='pollen',
        version=5,
    )

    history = pollen_fg.read()
    history = history[history["date"].dt.date < yesturday]

    context = history[history["date"].dt.date >(today-datetime.timedelta(days=100))].copy()

    sorted_context = context.sort_values("date").reset_index(drop=True)

    sorted_context['date'] = pd.to_datetime(sorted_context['date']).dt.date
    sorted_context['date'] = pd.to_datetime(sorted_context['date'])

    df_combined = pd.concat([sorted_context, df_pollen], ignore_index=True).sort_values("date").reset_index(drop=True)

    for pollen_type in pollen_types:

        lag1 = df_combined[pollen_type].shift(1)
        lag2 = df_combined[pollen_type].shift(2)
        lag3 = df_combined[pollen_type].shift(3)

        df_combined[f"{pollen_type}_lag1"] = lag1.ffill()
        df_combined[f"{pollen_type}_lag2"] = lag2.ffill()
        df_combined[f"{pollen_type}_lag3"] = lag3.ffill()

    df_to_insert = df_combined[df_combined["date"].dt.date == yesturday].copy()
    df_to_insert = df_to_insert.fillna(0.0)
    print(df_to_insert)
    print(df_to_insert.columns.tolist())
    print(df_to_insert.info())

    pollen_fg.insert(df_to_insert, wait=True)

def main():
    print("STARTING WITH WETHER INGEST: ")
    ingestWether()
    print("WETHER INGEST DONE.")
    print("INGESTING POLLEN: ")
    ingest_pollen()
    print("POLLEN INGEST DONE.")

    print("------------------------------------------------------ \n Done.")
    


if __name__ == "__main__":
    main()
