# Avalanche Incidents in Tirol
This repository contains a comprehensive dataset of historical avalanche incidents in the Tirol federal state, along with Jupyter notebooks detailing the methods used to fetch and preprocess this data. Additionally, you'll find a Jupyter notebook where available avalanche bulletins with incident data are compared, providing valuable insights into avalanche risk assessment. The dataset was also used to train different machine learning models and predict the avalanche danger level for incidents.

## Datasets
The datasets contain avalanche incidents from 1992 until March 2024 for the Austrian federal state of Tyrol, collected from the LAWIS database. For all incidents, relevant weather data was fetched and added to the dataset. They are stored as CSV files in the `data` folder. For more information about them, please refer to that folder.

[LAWIS](https://lawis.at/incident/) is used as the source of avalanche incidents, the [OpenMeteo API](https://open-meteo.com/) provides meteorological data, and [avalache.report](https://avalanche.report/more/archive) is used for avalanche bulletin data.

## Jupyter Notebooks
The entire process of creating the dataset is documented in the Jupyter notebooks located in the root folder of this repository. You can run the notebooks to fetch the latest data or create your dataset for a different region or time period.

[`01_creating_the_dataset.ipynb`](01_creating_the_dataset.ipynb) - Initially, incident data and meteorological data are fetched. Then, these two sources of data are merged into a single table, and unnecessary and potentially sensitive data are removed.

[`02_data_preprocessing.ipynb`](02_data_preprocessing.ipynb) - In the second step, the data undergoes preprocessing. During the exploratory data analysis, I identified various suspicious distributions and decided to remove them after further analysis. For the terrain features, I utilized the GDAL library to preprocess a Digital Elevation Model (DEM) of Tyrol, provided by the federal state as open data. A slope aspect and slope angle model were created and used to fill in missing terrain data. The notebook includes visualizations of feature distributions, correlation matrices, and interactive maps created with `folium`.


[`03_avalanche_bulletin_analysis.ipynb`](03_avalanche_bulletin_analysis.ipynb) - After preprocessing the dataset, historical avalanche bulletins can be fetched and compared with incident records. This comparison provides insight into the accuracy of the forecasts in the bulletins and their impact on avalanche incidents.

[`04_danger_rating_classification.ipynb`](04_danger_rating_classification.ipynb) - Onsite, avalanche danger ratings for incidents are assessed. Accurate prediction of avalanche danger ratings by trained machine learning classifiers could serve as a decision support tool for avalanche experts and could validate their bulletin forecasts based on historical events in the area. Further preprocessing was needed to use the dataset. Models used for performance comparison include random forest, XGBoost, SVC, and logistic regression.

## Project Structure
```bash
.
├── assets
│   ├── classification                      # trained classifiers
│   ├── geo_models                          # terrain models
│   └── images                              # static images
├── avalancheutils                          # Python module with utility functions
├── data                
│   ├── cache
│   │   ├── bulletins                       # fetched bulletins cache
│   │   ├── incidents                       # fetched incidents cache
│   │   └── weather                         # weather data cache
│   ├── incidents_tirol.csv
│   └── incidents_tirol_processed.csv
├── 01_creating_the_dataset.ipynb
├── 02_data_preprocessing.ipynb
├── 03_avalanche_bulletin_analysis.ipynb    
├── 04_danger_rating_classification.ipynb
├── config.yml
├── requirements.txt
└── setup.sh
```
* `assets` - Used for storing the downloaded DEM and other terrain models as well as trained classifiers. The `images` folder is used for static images in the notebooks.
* `avalancheutils` - Module with all the utility functions used in the notebooks for fetching data, preprocessing, classification, and visualization.
* `data` - This folder is where all the fetched data is stored. It also serves as a cache for fetched incident records, weather data, and avalanche bulletins.
* `01_creating_the_dataset.ipynb` - Fetching incident and weather data, merging it into a single table.
* `02_data_preprocessing.ipynb` - Contains exploratory data analysis, including distributions of values using histograms and maps. It removes invalid records and fills empty terrain feature values.
* `03_avalanche_bulletin_analysis.ipynb` - Compares data from historical avalanche bulletins with data from the preprocessed dataset, providing insights into avalanche risk assessment.
* `04_danger_rating_classification.ipynb` - Trains classification models to predict the avalanche danger rating for incidents.
* `config.yml` - Configuration file for the `avalancheutils` module; currently, it only contains configuration for additional `Folium.TileLayer`.
* `requirements.txt` - File listing the Python dependencies required for running the notebooks.
* `setup.sh` - Setup script for configuring the environment and installing necessary dependencies.
  
## Setup Guide
Before running the setup script, make sure that you have GDAL and Python development bindings installed. On Fedora you can install them with:
```bash
sudo dnf install gdal python3-devel -y
```
For other distributions or systems, see their installation guide [here](https://gdal.org/download.html).

On GNU/Linux you can setup use the setup script, that will create a Python virtual environment, download all the necessary packages and download and extract the DEM file that is used for data preprocessing. You can run it like this:
```bash
./setup.sh [-n]
```
The `-n` flag is optional and you can use it when you don't want to download the DEM file.

