# Datasets
## Raw Dataset (`incidents_tirol.csv`)
This dataset is the result of the process described in `01_creating_the_dataset.ipynb`. Essentially, it was created by merging all available incident records in the LAWIS system with meteorological data from OpenMeteo and removing unnecessary columns (`images`, `comments`, and metadata about the report). Only data for Tirol was filtered. For reproducibility purposes, the dataset only contains events that occurred before 2024-03-01. Check out the notebook yourself to fetch the newest data and create your version of the dataset.

### Dataset Structure:
* Contains 3060 records from 1992-11-17 until 2024-02-29.
* Consists of 41 columns containing 85589 non-empty values, approximately 68.2% of all the values.


| Column Name                      | Non-empty Values Count (Non-empty Percentage) | Data Type      | Description                                                                                                   |
| -------------------------------- | --------------------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------- |
| id                               | 3060 (100)                                    | int64          | Unique identifier of the incident                                                                             |
| date                             | 3060 (100)                                    | datetime64[ns] | Date and time of occurrence                                                                                   |
| valid_time                       | 3060 (100)                                    | bool           | Validity of the time of occurence                                                                             |
| location_longitude               | 3060 (100)                                    | float64        | Longitude (WGS84)                                                                                             |
| location_latitude                | 3060 (100)                                    | float64        | Latitude (WGS84)                                                                                              |
| danger_rating_level              | 2297 (75.07)                                  | float64        | Avalanche danger level [[source](https://avalanche.report/education/danger-scale)]                            |
| danger_rating_text               | 2323 (75.92)                                  | object (str)   | Text representation of the avalanche danger level [[source](https://avalanche.report/education/danger-scale)] |
| danger_problem_text              | 487 (15.92)                                   | object (str)   | Avalanche problem [source](https://avalanche.report/education/avalanche-problems)]                            |
| involved_sum                     | 775 (25.33)                                   | float64        | Number of individuals involved                                                                                |
| involved_dead                    | 2919 (95.39)                                  | float64        | Number of dead                                                                                                |
| involved_injured                 | 2731 (89.25)                                  | float64        | Number of injured                                                                                             |
| involved_uninjured               | 797 (26.05)                                   | float64        | Number of uninjured                                                                                           |
| involved_swept                   | 2613 (85.39)                                  | float64        | Number of swept                                                                                               |
| involved_buried_total            | 2648 (86.54)                                  | float64        | Number of fully buried                                                                                        |
| involved_buried_partial          | 2646 (86.47)                                  | float64        | Number of partially buried                                                                                    |
| involved_not_buried              | 195 (6.37)                                    | float64        | Number of not buried                                                                                          |
| avalanche_extent_length          | 1697 (55.46)                                  | float64        | Length of avalanche (m)                                                                                       |
| avalanche_extent_width           | 1678 (54.84)                                  | float64        | Width of avalanche at release point (m)                                                                       |
| avalanche_breakheight            | 1241 (40.56)                                  | float64        | Height of avalanche at release point (cm)                                                                     |
| avalanche_type_text              | 2474 (80.85)                                  | object (str)   | Avalanche type [[source](https://www.avalanches.org/glossary/#avalanche-types)]                               |
| avalanche_size_text              | 715 (23.37)                                   | object (str)   | Avalanche size [[source](https://avalanche.report/education/avalanche-sizes)]                                 |
| location_elevation               | 2337 (76.37)                                  | float64        | Elevation (m. a. s. l.)                                                                                       |
| location_slope_angle             | 1865 (60.95)                                  | float64        | Slope angle (˚)                                                                                               |
| location_aspect_text             | 2219 (72.52)                                  | object (str)   | Slope aspect as cardinal direction                                                                            |
| location_name                    | 3060 (100)                                    | object (str)   | Precise location name                                                                                         |
| location_subregion_text          | 3060 (100)                                    | object (str)   | Subregion name [[source](https://gitlab.com/eaws/eaws-regions)]                                               |
| involved_equipment_standard_text | 225 (7.35)                                    | object (str)   | Standard avalanche equipment availability in the group (transceiver, probe, shovel)                           |
| involved_equipment_lvs_text      | 205 (6.70)                                    | object (str)   | AVD activation                                                                                                |
| involved_equipment_airbag_text   | 173 (5.65)                                    | object (str)   | Availability of airbags in the group (all, some, none)                                                        |
| involved_ascent_descent_text     | 411 (13.43)                                   | object (str)   | Phase of the tour (ascent, descent, not Moving)                                                               |
| avalanche_release_text           | 492 (16.08)                                   | object (str)   | Cause of the avalanche release (artificial, spontaneous)                                                      |
| avalanche_humidity_text          | 466 (15.23)                                   | object (str)   | Avalanche humidity (dry, wet)                                                                                 |
| weather_temp                     | 3060 (100)                                    | float64        | Air temperature 2 meters above surface (˚C)                                                                   |
| weather_snow_depth               | 3060 (100)                                    | float64        | Snow depth (m)                                                                                                |
| weather_temp_mean                | 3060 (100)                                    | float64        | Mean temperature for the previous 3 days (˚C)                                                                 |
| weather_temp_diff                | 3060 (100)                                    | float64        | Temperature difference for the previous 3 days (˚C)                                                           |
| weather_rain_sum                 | 3060 (100)                                    | float64        | Rainfall for the previous 3 days (˚C)                                                                         |
| weather_snow_sum                 | 3060 (100)                                    | float64        | Snowfall for the previous 3 days (˚C)                                                                         |
| weather_wind_speed_mean          | 3060 (100)                                    | float64        | Mean maximum wind speed for the previous 3 days (m/s)                                                         |
| weather_wind_dir_mean            | 3060 (100)                                    | float64        | Mean wind direction for the previous 3 days as azimuth (˚)                                                    |
| weather_radiation_sum            | 3060 (100)                                    | float64        | Shortwave solar radiation for the previous 3 days (MJ/m<sup>2</sup>)                                          |


## Preprocessed Dataset (`incidents_tirol_processed.csv`)
This dataset is the result of the process described in `02_data_preprocessing.ipynb`. During the process, an exploratory data analysis (EDA) was conducted, and I cleaned the data and filled missing values using a Digital Elevation Model (DEM) file. A large number of records were clustered to a few points in Tyrol, which I decided to remove. Their removal also helped to equalize the yearly event distribution. The resulting dataset is further used in avalanche bulletin analysis and danger rating classification.

### Changes Summary
* Corrected the `valid_time` values to accurately reflect the time validity.
* Added a new column `week_day` containing the day number in the week.
* Removed records with invalid `location_longitude` and `location_latitude`.
* Utilized DEM to fill missing terrain data (`location_elevation`, `location_slope_angle`, and `location_aspect`) and removed records where the computed and recorded values differed significantly.
* Removed the redundant `danger_rating_level` column.
* Removed redundant `_text` suffixes in column names
### Dataset Structure:
* Contains 1109 records from 1993-01-27 until 2024-02-29.
* 41 columns containing 36161 non-empty values, approximately 79.5% of all the values.

| Column Name                  | Non-empty Values Count (Non-empty Percentage) | pandas dtype   | Description                                                                           |
| ---------------------------- | --------------------------------------------- | -------------- | ------------------------------------------------------------------------------------- |
| id                           | 1109 (100)                                    | int64          | Unique identifier of the incident                                                     |
| date                         | 1109 (100)                                    | datetime64[ns] | Date and time of occurrence                                                           |
| valid_time                   | 1109 (100)                                    | bool           | Validity of the time record                                                           |
| location_longitude           | 1109 (100)                                    | float64        | Longitude (WGS84)                                                                     |
| location_latitude            | 1109 (100)                                    | float64        | Latitude (WGS84)                                                                      |
| danger_rating                | 1054 (95.04)                                  | float64        | Avalanche danger level [[source](https://avalanche.report/education/danger-scale)]    |
| danger_problem               | 478 (43.10)                                   | object (str)   | Avalanche problem [[source](https://avalanche.report/education/avalanche-problems)]   |
| involved_sum                 | 750 (67.63)                                   | float64        | Number of individuals involved                                                        |
| involved_dead                | 974 (87.83)                                   | float64        | Number of dead                                                                        |
| involved_injured             | 976 (88.01)                                   | float64        | Number of injured                                                                     |
| involved_uninjured           | 772 (69.61)                                   | float64        | Number of uninjured                                                                   |
| involved_swept               | 970 (87.47)                                   | float64        | Number of swept                                                                       |
| involved_buried_total        | 963 (86.83)                                   | float64        | Number of fully buried                                                                |
| involved_buried_partial      | 965 (87.02)                                   | float64        | Number of partially buried                                                            |
| involved_not_buried          | 192 (17.31)                                   | float64        | Number of not buried                                                                  |
| avalanche_extent_length      | 817 (73.67)                                   | float64        | Length of avalanche (m)                                                               |
| avalanche_extent_width       | 802 (72.32)                                   | float64        | Width of avalanche at release point (m)                                               |
| avalanche_breakheight        | 640 (57.71)                                   | float64        | Height of avalanche at release point (cm)                                             |
| avalanche_type               | 1000 (90.17)                                  | object (str)   | Avalanche type [[source](https://www.avalanches.org/glossary/#avalanche-types)]       |
| avalanche_size               | 695 (62.67)                                   | float64        | Avalanche size [[source](https://avalanche.report/education/avalanche-sizes)]         |
| location_elevation           | 1109 (100)                                    | float64        | Elevation (m above sea level)                                                         |
| location_slope_angle         | 1109 (100)                                    | float64        | Slope angle (˚)                                                                       |
| location_aspect              | 1109 (100)                                    | object (str)   | Slope aspect as cardinal direction                                                    |
| location_name                | 1109 (100)                                    | object (str)   | Precise location name                                                                 |
| location_subregion           | 1109 (100)                                    | object (str)   | Subregion name [[source](https://gitlab.com/eaws/eaws-regions)]                       |
| involved_standard_equipment  | 220 (19.84)                                   | object (bool)  | Whether the individuals had standard avalanche equipment (transceiver, probe, shovel) |
| involved_avd_activated       | 200 (18.03)                                   | object (bool)  | Whether the AVD was activated                                                         |
| involved_equipment_airbag    | 168 (15.15)                                   | object (str)   | Availability of airbags in the group (all, some, none)                                |
| involved_ascent_descent      | 404 (36.43)                                   | object (str)   | Phase of the tour (ascent, descent, stationary)                                       |
| avalanche_artificial_release | 483 (43.55)                                   | object (bool)  | Whether the avalanche was released artificially                                       |
| avalanche_humidity           | 458 (41.30)                                   | object (bool)  | Whether the avalanche was wet                                                         |
| weather_temp                 | 1109 (100)                                    | float64        | Air temperature 2 meters above surface (˚C)                                           |
| weather_snow_depth           | 1109 (100)                                    | float64        | Snow depth (m)                                                                        |
| weather_temp_mean            | 1109 (100)                                    | float64        | Mean temperature for the previous 3 days (˚C)                                         |
| weather_temp_diff            | 1109 (100)                                    | float64        | Temperature difference for the previous 3 days (˚C)                                   |
| weather_rain_sum             | 1109 (100)                                    | float64        | Rainfall for the previous 3 days (˚C)                                                 |
| weather_snow_sum             | 1109 (100)                                    | float64        | Snowfall for the previous 3 days (˚C)                                                 |
| weather_wind_speed_mean      | 1109 (100)                                    | float64        | Mean maximum wind speed for the previous 3 days (m/s)                                 |
| weather_wind_dir_mean        | 1109 (100)                                    | float64        | Mean wind direction for the previous 3 days as azimuth (˚)                            |
| weather_radiation_sum        | 1109 (100)                                    | float64        | Shortwave solar radiation for the previous 3 days (MJ/m<sup>2</sup>)                  |
| week_day                     | 1109 (100)                                    | int32          | Day of the week (indexed from Monday - 0)                                             |
