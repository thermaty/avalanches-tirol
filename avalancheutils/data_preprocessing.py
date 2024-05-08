from datetime import time
from math import floor
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder

pd.set_option('future.no_silent_downcasting', True)


def decimal_to_hh_mm(dec: float):
    """
    Convert hour and time represented as a decimal number to hours and minutes.

    :param dec: Decimal number representing time.
    :return: Time object representing the converted hours and minutes.
    """
    hours, minutes = floor(dec), int((dec % 1) * 60)
    return time(hour=hours, minute=minutes)


def iqr_outlier_detection(data: pd.DataFrame | pd.Series):
    """
    Detect outliers in the data using the interquartile range method.

    :param data: DataFrame or Series containing the data.
    :return: Outliers detected in the data.
    """
    q_1, q_3 = data.quantile(0.25), data.quantile(0.75)
    IQR = q_3 - q_1
    lower_bound, upper_bound = q_1 - 1.5 * IQR, q_3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers


def unique_value_counts(col: pd.Series, title: str = '', description: bool = True):
    """
    Count unique values in a pandas Series and optionally print a description.

    :param col: Series containing the data.
    :param title: Optional. Title for the property (default: '').
    :param description: Optional. Whether to print a description (default: True).
    :return: Series containing unique value counts.
    """
    unique_values = col.value_counts().sort_values(ascending=False)
    if description:
        print(
            f'Unique values for the property "{title}" ({len(unique_values)} unique values and {col.count()} / {len(col)} '
            f'non-null values):\n')
    return unique_values


def adjust_column(df: pd.DataFrame, col_name: str, value_mapping: dict, rename: str | None = None):
    """
    Adjust values in a DataFrame column based on a mapping dictionary and optionally rename the column.

    :param df: DataFrame containing the data.
    :param col_name: Name of the column to adjust.
    :param value_mapping: Dictionary with the value mapping.
    :param rename: Optional. New name for the column (default: None).
    :return: DataFrame with adjusted column values.
    """
    df[col_name] = df[col_name].replace(value_mapping).infer_objects(copy=False)
    if rename:
        df = df.rename(columns={col_name: rename})
    return df


def compare_column_operation(before: pd.Series, after: pd.Series, description: str = None, diff: bool = False):
    """
    Compare values between two Series and optionally calculate differences.

    :param before: Series containing values before an operation.
    :param after: Series containing values after an operation.
    :param description: Optional. Description of the comparison (default: None).
    :param diff: Optional. Whether to calculate absolute differences (default: False).
    :return: DataFrame containing the comparison results.
    """
    describe_comparison = pd.concat([before, after], axis=1)
    column_names = ['Before', 'After']
    if diff:
        describe_comparison['Absolute Difference'] = abs(
            describe_comparison.iloc[:, 1] - describe_comparison.iloc[:, 0])
        column_names.append('Absolute Difference')
    describe_comparison.columns = column_names
    if description:
        print(description)
    return describe_comparison


def degrees_to_text(degree_val: float) -> str:
    """
    Convert an azimuth value in degrees to a cardinal direction ('N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW').
    inspired by: https://stackoverflow.com/a/7490772

    :param degree_val: Azimuth in degrees.
    :return: str representation of the corresponding cardinal direction.
    """
    if degree_val < 0:
        raise ValueError(f"Input value was {degree_val} and it should be positive")
    idx = int((degree_val / 45) + .5)
    cardinal_directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    return cardinal_directions[(idx % 8)]


def angle_difference(x: float, y: float):
    """
    Calculate the difference between two angles.

    :param x: First angle.
    :param y: Second angle.
    :return: Angle difference.
    """
    if pd.isna(x) or pd.isna(y):
        return np.nan
    diff = abs(x - y)
    if diff > 180:
        diff = 360 - diff
    return diff


def data_overview(data: pd.DataFrame, include_dtype=False):
    """
    Generate an overview of the DataFrame.

    :param data: The DataFrame for which to generate the overview.
    :param include_dtype: Whether to include pandas data types. Defaults to False.
    :return: A DataFrame containing the count of unique values, non-null values, and optionally, data types.
    """
    result = pd.DataFrame({'Unique Values': data.nunique(), 'Non-Null Values': data.count()})
    if include_dtype:
        result['Data Type'] = data.dtypes
    return result.sort_values(by='Non-Null Values')


def one_hot_encode(data: pd.DataFrame, columns_to_encode: Iterable[str] | pd.Index):
    """
    One-hot encode specified categorical columns in a DataFrame.

    :param data: DataFrame containing the data to encode.
    :param columns_to_encode: Iterable of column names or indices to one-hot encode.
    :return: DataFrame containing the one-hot encoded data.
    """
    one_hot_encoder = OneHotEncoder()
    transformer = ColumnTransformer([('one-hot', one_hot_encoder, columns_to_encode)], remainder='passthrough',
                                    verbose_feature_names_out=False)
    encoded_incidents = transformer.fit_transform(data)
    return pd.DataFrame(encoded_incidents, columns=transformer.get_feature_names_out())


def missing_data_overview(data: pd.DataFrame, threshold: float = 101, missing_cols_idx: bool = False):
    """
    Generate an overview of missing data in a DataFrame.

    :param data: DataFrame for which to generate the overview.
    :param threshold: Threshold percentage for considering columns with missing data (default: 101).
    :param missing_cols_idx: Whether to return the index of columns with missing data (default: False).
    :return: DataFrame containing the overview of missing data and optionally the index of columns with missing data.
    """
    non_null_percentages = (data.count() / len(data)) * 100
    columns_below_threshold = non_null_percentages[non_null_percentages < threshold].sort_values(ascending=False)
    columns_below_threshold.name = 'Non-Null Values (%)'
    observed_columns = data[columns_below_threshold.index]
    overview = pd.concat([columns_below_threshold, data_overview(observed_columns)], axis=1).rename_axis('Feature Name',
                                                                                                         axis='index')
    if missing_cols_idx:
        return overview, columns_below_threshold.index
    return overview


def adjust_scaled_data(data: pd.DataFrame, scaler, columns_to_round: Iterable[str] | pd.Index = None):
    """
    Adjust scaled data by applying inverse transformation and optionally rounding specific columns.

    :param data: DataFrame containing the scaled data.
    :param scaler: Scaler object used for scaling.
    :param columns_to_round: Optional. Iterable of column names or indices to round.
    :return: DataFrame containing the adjusted data.
    """
    res = pd.DataFrame(scaler.inverse_transform(data), columns=data.columns)
    if columns_to_round:
        res[columns_to_round] = res[columns_to_round].round()
    return res


def impute_categorical_feature(data, categorical_feature):
    """
    Directly impute null values in a categorical feature using sklearn.HistGradientBoostingClassifier.

    :param data: The input DataFrame containing the dataset.
    :param categorical_feature: The name of the categorical feature to be imputed.
    """
    # Split the dataset into features and the target (the column to be imputed)
    X = data.drop(columns=[categorical_feature])
    y = data[categorical_feature]

    # Create a boolean mask to identify rows with missing values
    y_nan_values = y.isnull()

    # Split data into training and prediction subsets
    # Training data: rows with non-null values
    # Prediction data: rows with null values
    X_train, y_train = X[~y_nan_values], y[~y_nan_values]
    X_pred = X[y_nan_values]

    # Train the classifier on the non-null data
    classifier = HistGradientBoostingClassifier(l2_regularization=0.1)
    classifier.fit(X_train, y_train)
    # Use the trained classifier to predict missing values
    data.loc[y_nan_values, categorical_feature] = classifier.predict(X_pred)


def angle_decomposition(angles) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose angles into their x and y components on the unit circle.

    :param angles: Array-like object containing angles in degrees.
    :return: Tuple of arrays containing the x and y components.
    """
    x = np.cos(np.radians(angles))
    y = np.sin(np.radians(angles))
    return x, y
