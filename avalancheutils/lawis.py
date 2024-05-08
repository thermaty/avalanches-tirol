import json
from datetime import datetime
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests
from requests import Session
from retry_requests import retry

rng = np.random.default_rng()


def fetch_event_overview(start_date="1970-01-01", end_date=datetime.today().strftime('%Y-%m-%d'), output_dir='data',
                         save=True, api="incident"):
    headers = {'accept': 'application/json'}
    try:
        response = requests.get(f'https://lawis.at/lawis_api/v2_2/{api}/?startDate={start_date}&endDate={end_date}',
                                headers=headers)
    except requests.exceptions.RequestException as e:
        print(f'An Exception occurred while fetching the overview: {e}')
    else:
        json_response = json.loads(response.text)
        if save:
            with (Path(output_dir) / f'{api}_overview.json').open('w') as output:
                json.dump(json_response, output)
        res = pd.DataFrame(json_response)
        print(f'Fetched an event overview containing {res.shape[0]} events.')
        return res


def fetch_events_details(event_ids, api, output_dir):
    session = retry(Session(), retries=5, backoff_factor=0.3)
    success = 0
    new_events = 0
    for event_id in event_ids:
        event_file = f'{event_id:05d}.json'
        # Check if the record already exists
        if not (Path(output_dir) / event_file).exists():
            new_events += 1
            print(f'Fetching event details for event with id: {event_id}')
            try:
                response = session.get(
                    f'https://lawis.at/lawis_api/public/{api}/{event_id}',
                    headers={"accept": "application/json"})
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                print(f"for the event id: {event_id} an exception occurred: {e}")
            else:
                success += 1
                with (Path(output_dir) / event_file).open('w') as output:
                    json.dump(json.loads(response.text), output)
                sleep(rng.uniform(0.05, 0.2))
    if new_events > 0:
        print(f"Successfully fetched {success} events out of {new_events} new events.")
    return load_events_to_dataframe(output_dir)


def load_events_to_dataframe(data_folder):
    data = []
    for file_path in Path(data_folder).glob('*.json'):
        if file_path.is_file():
            with file_path.open('r') as file:
                row = json.load(file)
                data.append(row)
    return pd.json_normalize(data, sep='_')
