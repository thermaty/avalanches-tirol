from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from openmeteo_sdk.VariablesWithTime import VariablesWithTime
from retry_requests import retry
from scipy.stats import circmean

# Initialize the cache and retry_requests session
cache_session = requests_cache.CachedSession('data/cache/weather/weather_cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)

# OpenMeteo is used as the provider of the weather data
openmeteo = openmeteo_requests.Client(session=retry_session)
url = "https://archive-api.open-meteo.com/v1/archive"
rng = np.random.default_rng()


def get_weather_data(lon: float, lat: float, start_date: str, end_date: str):
    """
    Retrieve OpenMeteo hourly and daily weather data for a given location and time period.

    :param lon: Longitude of the location.
    :param lat: Latitude of the location.
    :param start_date: Start date of the weather data retrieval in 'YYYY-MM-DD' format.
    :param end_date: End date of the weather data retrieval in 'YYYY-MM-DD' format.
    :return: A tuple containing hourly and daily weather data as pandas DataFrames.
    """
    hourly_variables = ["temperature_2m", "snow_depth"]
    daily_variables = ["temperature_2m_mean", "rain_sum", "snowfall_sum", "wind_speed_10m_max",
                       "wind_direction_10m_dominant", "shortwave_radiation_sum"]

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_variables,
        "daily": daily_variables,
        "wind_speed_unit": "ms",
        "models": "best_match"
    }
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    hourly_data = _response_to_dict(response.Hourly(), hourly_variables)
    daily_data = _response_to_dict(response.Daily(), daily_variables)
    return pd.DataFrame(data=hourly_data), pd.DataFrame(data=daily_data)


def _response_to_dict(response: VariablesWithTime, variables: List[str]):
    """
    Convert OpenMeteo API response to a dictionary containing variable values.

    :param response: OpenMeteo API response object.
    :param variables: List of variable names.
    :return: A dictionary containing variable values with dates as keys.
    """
    variable_values = [response.Variables(i).ValuesAsNumpy() for i in range(len(variables))]
    return {"date": pd.date_range(
        start=pd.to_datetime(response.Time(), unit="s"),
        end=pd.to_datetime(response.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=response.Interval()),
        inclusive="left"), **dict(zip(variables, variable_values))}


def get_weather_for_incident(incident: pd.Series):
    """
    Retrieve weather data for a given incident.

    :param incident: An incident record as a pandas Series.
    :return: A dictionary containing weather report for the incident.
    """
    report = {'id': incident['id'], 'temp': np.nan, 'snow_depth': np.nan, 'temp_mean': np.nan,
              'temp_diff': np.nan, 'rain_sum': np.nan, 'snow_sum': np.nan,
              'wind_speed_mean': np.nan, 'wind_dir_mean': np.nan, 'radiation_sum': np.nan}
    start_date = (incident['date'] - pd.Timedelta(days=3)).strftime('%Y-%m-%d')
    end_date = incident['date'].strftime('%Y-%m-%d')
    hourly, daily = get_weather_data(lon=incident['location_longitude'], lat=incident['location_latitude'],
                                     start_date=start_date,
                                     end_date=end_date)
    incident_date_hour = incident['date'].floor('h')
    incident_hour_weather = hourly[hourly['date'] == incident_date_hour]
    report['temp'] = incident_hour_weather['temperature_2m'].iloc[0]
    report['snow_depth'] = incident_hour_weather['snow_depth'].iloc[0]
    report['temp_mean'] = daily['temperature_2m_mean'].mean()
    report['temp_diff'] = daily['temperature_2m_mean'].iloc[-1] - daily['temperature_2m_mean'].iloc[0]
    report['rain_sum'] = daily['rain_sum'].sum()
    report['snow_sum'] = daily['snowfall_sum'].sum()
    report['wind_speed_mean'] = daily['wind_speed_10m_max'].mean()
    report['wind_dir_mean'] = circular_mean(daily['wind_direction_10m_dominant'].values)
    report['radiation_sum'] = daily['shortwave_radiation_sum'].sum()
    return report


def fetch_weather_data(incidents, weather_file):
    """
    Fetch weather data for incidents and save it to a CSV file.

    :param incidents: DataFrame containing incident records.
    :param weather_file: Path to the CSV file storing weather data.
    :return: DataFrame containing weather data for incidents.
    """
    processed_incidents = incidents
    result = pd.DataFrame()
    success = 0
    # check if the weather data was already obtained
    if Path(weather_file).exists():
        result = pd.read_csv(weather_file)
        # fetch weather data for new incidents only
        processed_incidents = incidents[~incidents['id'].isin(result['id'])]
    incident_weather = []
    for idx, incident in processed_incidents.iterrows():
        sleep(rng.uniform(0.01, 0.2))
        incident_weather.append(get_weather_for_incident(incident))
        success += 1
        print(f'Fetching weather data for the incident with id: {incident['id']}')
    result = pd.concat([result, pd.DataFrame(incident_weather)])
    result.to_csv(weather_file, index=False)
    if success > 0:
        print(f"Successfully fetched {success} events out of {len(processed_incidents)} new events.")
    return result


def circular_mean(degree_values):
    """
    Compute the circular mean of a set of degree values.

    :param degree_values: An array-like object containing degree values.
    :return: The circular mean of the input degree values.
    """
    direction_radians = np.deg2rad(degree_values)
    return np.rad2deg(circmean(direction_radians))
