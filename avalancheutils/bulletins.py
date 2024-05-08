import json
from datetime import datetime, timedelta
from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests import Session
from retry_requests import retry


def _get_possible_dates(start_date: str, end_date: str):
    """
    Generate possible bulletin dates between start_date and end_date while taking into consideration the avalanche
    season.

    :param start_date: The start date in the format 'YYYY-MM-DD'.
    :param end_date: The end date in the format 'YYYY-MM-DD'.
    :return: A list of possible bulletin dates in 'YYYY-MM-DD' format.
    """
    bulletin_season = [11, 12, 1, 2, 3, 4, 5]
    possible_dates = []
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    current_date = start_date
    while current_date <= end_date:
        if current_date.month in bulletin_season:
            possible_dates.append(current_date.strftime('%Y-%m-%d'))
        current_date += timedelta(days=1)
    return possible_dates


def fetch_bulletins(cache_dir, start_date: str = '2018-11-01',
                    end_date: str = (datetime.today() + timedelta(days=1)).strftime('%Y-%m-%d')):
    """
    Fetch avalanche bulletins within a specified date range and save them in JSON format.

    :param cache_dir: Directory where the bulletins will be saved.
    :param start_date: Start date of the range (format: 'YYYY-MM-DD') (default: '2018-11-01').
    :param end_date: End date of the range (format: 'YYYY-MM-DD') (default: current date + 1 day).
    :return: Dataframe containing the fetched bulletins.
    """
    possible_dates = _get_possible_dates(start_date, end_date)
    new_bulletins = []
    session = retry(Session(), retries=5, backoff_factor=0.3)
    for date in possible_dates:
        bulletin_url = f'{date}_en_CAAMLv6.json'
        bulletin_file = date + '.json'
        if not (Path(cache_dir) / bulletin_file).exists():  # check if the bulletin is already cached
            try:
                response = session.get(f'https://static.avalanche.report/bulletins/{date}/{bulletin_url}')
            except requests.exceptions.RequestException as e:
                print(f'for the bulletin: {bulletin_url} an exception occurred: {e}')
            else:
                if response.status_code == 200:  # check for HTTP OK, meaning the bulletin exists
                    print(f'Successfully fetched avalanche bulletin for the date: {date}')
                    new_bulletins.append(date)
                    with (Path(cache_dir) / bulletin_file).open('w') as output:
                        json.dump(json.loads(response.text), output)
    print(f'{len(new_bulletins)} new bulletins were fetched')
    return load_bulletins_dataframe(cache_dir)


def load_bulletins_dataframe(data_folder, output_file: str = None):
    """
    Load JSON avalanche bulletins files into a pandas DataFrame.

    :param data_folder: Directory containing the cached JSON files of the bulletins.
    :param output_file: Optional. If specified, saves the DataFrame as a CSV file named with .
    :return: DataFrame containing the loaded bulletins.
    """
    data = {}
    for file_path in Path(data_folder).glob('*.json'):  # only look for .json files
        if file_path.is_file():
            with file_path.open('r') as file:
                try:
                    row = json.load(file)
                except JSONDecodeError as e:
                    print(f'{file_path}: {e}')
                else:
                    data[file_path.stem] = row  # file_path.stem provides the bulletin date
    result = pd.DataFrame.from_dict(data, orient='index')
    result = result.set_index(pd.to_datetime(result.index), verify_integrity=True)  # set the date to be the index
    if output_file:
        result.to_csv(output_file, index_label='date')
    return result


def get_bulletin_data_for_incidents(incidents: pd.DataFrame, bulletins: pd.DataFrame):
    """
    Get bulletin data corresponding to incidents from incidents DataFrame.

    :param incidents: DataFrame containing incident data.
    :param bulletins: DataFrame containing bulletin data.
    :return: DataFrame containing bulletin data corresponding to incidents.
    """
    report_data = []
    for idx, incident in incidents.iterrows():
        report_data.append(_incident_bulletin_comparison(incident=incident, bulletin_data=bulletins))
    return pd.DataFrame(report_data)


def _incident_bulletin_comparison(incident: pd.Series, bulletin_data: pd.DataFrame):
    """
    Compare incident data with bulletin data and return comparison results.

    :param incident: Series containing an incident record.
    :param bulletin_data: DataFrame containing bulletin data.
    :return: Dictionary with the results of the comparison.
    """
    comparison = {'id': incident['id'], 'bulletin_exists': None, 'region_code': None,
                  'danger_rating': 0, 'danger_problem': np.nan}
    try:
        bulletins = bulletin_data.loc[pd.Timestamp(incident['date'].date())]['bulletins']
    except KeyError:
        pass  # there is no bulletin for the selected date, just return the default comparison
    else:
        comparison['bulletin_exists'] = True
        if bulletin := get_bulletin_by_region_code(bulletins, incident['region_code']):
            comparison['region_code'] = True
            incident_hour = incident['date'].hour if incident['valid_time'] is True else None
            incident_elevation = incident['location_elevation']
            comparison['danger_rating'] = _get_danger_rating(incident_time=incident_hour,
                                                             incident_elevation=incident_elevation, bulletin=bulletin)
            comparison['danger_problem'] = _get_avalanche_problems(incident_time=incident_hour,
                                                                   incident_elevation=incident_elevation,
                                                                   incident_aspect=incident['location_aspect'],
                                                                   bulletin=bulletin)
    return comparison


def get_bulletin_by_region_code(possible_bulletins, region_code):
    """
    Retrieve bulletin by region code from a list of possible bulletins.

    :param possible_bulletins: List of possible bulletins for the day.
    :param region_code: EAWS region code to match.
    :return: Bulletin corresponding to the provided region code, or None if not found.
    """
    for bulletin in possible_bulletins:
        for region in bulletin['regions']:
            if region['regionID'] == region_code:
                return bulletin
    return None


def _get_danger_rating(incident_time, incident_elevation: float, bulletin: dict):
    """
    Get the danger rating from the bulletin for an incident based on incident time and elevation.

    :param incident_time: Hour of the incident.
    :param incident_elevation: Elevation of the incident location (location_elevation).
    :param bulletin: Avalanche bulletin for the incident region.
    :return: Forecasted anger rating for the incident, or NaN if not found.
    """
    result = np.nan
    for danger_rating in bulletin['dangerRatings']:
        if _check_time_validity(incident_time, bulletin_time_period=danger_rating['validTimePeriod']):
            try:
                danger_elevation = danger_rating['elevation']
            except KeyError:
                return convert_danger_rating_text(danger_rating['mainValue'])
            else:
                elevation = check_elevation_bounds(incident_elevation=incident_elevation,
                                                   bulletin_elevation=danger_elevation)
                if elevation is True:
                    return convert_danger_rating_text(danger_rating['mainValue'])
    return result


def _get_avalanche_problems(incident_time, incident_elevation: float, incident_aspect: str, bulletin: dict):
    """
    Get avalanche problems from the bulletin for an incident based on incident time, elevation and aspect.

    :param incident_time: Hour of the incident.
    :param incident_elevation: Elevation of the incident location (location_elevation).
    :param incident_aspect: Aspect of the incident location (location_aspect).
    :param bulletin: Avalanche bulletin for the incident region.
    :return: List of dictionaries containing avalanche problem specifications (name and elevation range)
            corresponding to the incident, or NaN if no matching problem is found for the criteria.
    """
    result = []
    for avalanche_problem in bulletin['avalancheProblems']:
        if _check_time_validity(incident_time, bulletin_time_period=avalanche_problem['validTimePeriod']):
            if incident_aspect in avalanche_problem['aspects']:
                avalanche_problems_specs = {'danger_problem': avalanche_problem['problemType']}
                if (size := avalanche_problem.get('avalancheSize')) is not None:
                    avalanche_problems_specs['size'] = size
                try:
                    problem_elevation = avalanche_problem['elevation']
                except KeyError:
                    # if there is no elevation df in the bulletin, the avalanche problem is valid in all elevations
                    result.append(avalanche_problems_specs)
                else:
                    elevation = check_elevation_bounds(incident_elevation=incident_elevation,
                                                       bulletin_elevation=problem_elevation)
                    if elevation is True:
                        result.append(avalanche_problems_specs)
                    elif elevation is not False:
                        avalanche_problems_specs['elevation'] = elevation
                        result.append(avalanche_problems_specs)

    if len(result) == 0:
        return np.nan
    return result


def _convert_hour_to_valid_time_period(hour):
    """
    Convert an hour value to a validTimePeriod representation ('earlier' or 'later').

    :param hour: Hour value to convert. :return: Valid time period ('earlier' if hour is before noon,
                 'later' otherwise), or None if hour passed hour is None.
    """
    if hour is None:
        return None
    return 'earlier' if hour < 12 else 'later'


def _check_time_validity(incident_hour, bulletin_time_period):
    """
    Check the validity of incident hour against bulletin time period.

    :param incident_hour: Hour of the incident.
    :param bulletin_time_period: validTimePeriod specified in the bulletin.
    :return: True if the incident hour matches the bulletin time period, False otherwise.
    """
    time_period = _convert_hour_to_valid_time_period(incident_hour)
    if bulletin_time_period == 'all_day':
        return True
    if time_period is not None:
        return bulletin_time_period == time_period
    return False


def _add_missing_elevation_bounds(elevation):
    """
    Add missing elevation bounds to the provided elevation range dictionary.
    Missing 'lowerBound' is filled with 0, missing 'upperBound' with 9000.

    :param elevation: Dictionary containing elevation range bounds.
    :return: Updated elevation dictionary with missing bounds filled in.
    """
    elevation_bounds = elevation.copy()
    default_bounds = {"lowerBound": 0, "upperBound": 9000}
    if len(elevation) == 2:
        return elevation
    for key in default_bounds:
        if key not in elevation_bounds:
            elevation_bounds[key] = default_bounds[key]
    return elevation_bounds


def convert_danger_rating_text(danger_string):
    """
    Convert danger rating text to numeric value.

    :param danger_string: String representation of danger rating.
    :return: Numeric value corresponding to the danger rating text.
    """
    danger_ratings = {'low': 1,
                      'moderate': 2,
                      'considerable': 3,
                      'high': 4,
                      'very_high': 5}
    return danger_ratings[danger_string]


def check_elevation_bounds(incident_elevation, bulletin_elevation):
    """
    Check if incident elevation falls within the bounds specified in the bulletin elevation.

    :param incident_elevation: Elevation of the incident location.
    :param bulletin_elevation: Elevation bounds specified in the bulletin.
    :return: True if incident elevation is within the specified bounds, False otherwise.
    """
    elevation_range = _add_missing_elevation_bounds(bulletin_elevation)
    if 'treeline' in elevation_range.values():
        return [elevation_range['lowerBound'], elevation_range['upperBound']]
    return int(elevation_range['lowerBound']) <= int(incident_elevation) <= int(elevation_range['upperBound'])


def _replace_treeline_range(elevation_range: list, treeline_value: float):
    """
    Replace 'treeline' values in elevation range with the specified treeline value.

    :param elevation_range: List containing elevation bounds.
    :param treeline_value: Value to replace 'treeline' with.
    :return: List with 'treeline' replaced by the specified value.
    """
    return [int(bound) if bound != 'treeline' else treeline_value for bound in elevation_range]


def check_matching_danger_problem_size(incident, treeline_elevation=None, purge_treeline=True):
    """
    Check if the danger problem in the incident matches with any of the bulletin avalanche problems
    and return the corresponding avalanche size.

    :param incident: Series containing incident data.
    :param treeline_elevation: Elevation value to use for 'treeline' in elevation range (default: None).
    :param purge_treeline: Whether to exclude 'treeline' elevation range from consideration (default: True).
    :return: Dictionary containing the matching danger problem and avalanche size, or NaN if no match is found.
    """

    danger_problem = incident['danger_problem']
    possible_problems = incident['danger_problem_bulletin']
    result = {'danger_problem': np.nan, 'avalanche_size': np.nan}
    if possible_problems is np.nan or pd.isna(danger_problem):
        return result
    if purge_treeline:
        possible_problems = [problem for problem in possible_problems if problem.get('elevation') is None]
        if len(possible_problems) == 0:
            return result
        for problem in possible_problems:
            if danger_problem == problem['danger_problem']:
                result['danger_problem'] = danger_problem
                result['avalanche_size'] = problem.get('size', np.nan)
                return result
    else:
        for problem in possible_problems:
            elevation_range = problem.get('elevation')
            if elevation_range is None:
                if problem['danger_problem'] == danger_problem:
                    result['danger_problem'] = danger_problem
                    result['avalanche_size'] = problem.get('size', np.nan)
                    return result
            elif len(elevation_range) == 2:
                elevation_range = _replace_treeline_range(elevation_range=elevation_range,
                                                          treeline_value=treeline_elevation)
                if elevation_range[0] <= int(incident['location_elevation']) <= elevation_range[1]:
                    result['danger_problem'] = danger_problem
                    result['avalanche_size'] = problem.get('size', np.nan)
                    return result
    result['danger_problem'] = False
    return result
