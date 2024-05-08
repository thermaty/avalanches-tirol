import time
import warnings
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
from sklearn.metrics import balanced_accuracy_score, classification_report, cohen_kappa_score, f1_score, \
    precision_score, \
    recall_score
from sklearn.model_selection import GridSearchCV


def calculate_performance_metrics(y_true, y_pred, class_metrics=False):
    """
    Calculate performance metrics for a classification task. Currently weighted precision, weighted F1-Score,
    weighted recall, Cohen Kappa Score and balanced accuracy are computed.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param class_metrics: Whether to return per class metrics as well or not.
    :return: DataFrame containing overall performance metrics. If class_metrics is True, also returns a DataFrame
             containing class-specific metrics.
    """
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=np.nan)
    kappa = cohen_kappa_score(y_true, y_pred)
    bal_accuracy = balanced_accuracy_score(y_true, y_pred)
    performance_df = pd.DataFrame({
        'Metric': ['F1-score', 'Precision', 'Recall', 'Cohen Kappa Score', 'Balanced Accuracy'],
        'Value': [f1, precision, recall, kappa, bal_accuracy]
    })
    if class_metrics:
        class_report = classification_report(y_true, y_pred, zero_division=np.nan, output_dict=True)
        class_metrics_df = pd.DataFrame(class_report).T.iloc[:-3]
        return performance_df, class_metrics_df
    return performance_df


def get_model_predict_comparison(models: dict, X_test: Union[pd.DataFrame, np.ndarray],
                                 y_test: Union[pd.Series, np.ndarray]):
    """
    Compare the predictions of multiple models against the true labels and calculate performance metrics.

    :param models: A dictionary where keys are model names and values are dictionaries containing model objects
                   and optional scalers.
    :param X_test: Test features.
    :param y_test: True labels for the test data.
    :return: A tuple containing a dictionary with model names as keys and their corresponding predictions as values,
             and a DataFrame with performance metrics for each model.
    """
    results = []
    y_pred_dict = {}
    for name, model_params in models.items():
        model = model_params['model']
        scaler = model_params.get('scaler', None)
        y_pred = scale_predict(model, X_test, scaler)
        model_metrics = calculate_performance_metrics(y_test, y_pred)
        y_pred_dict[name] = y_pred
        results.append(model_metrics.set_index('Metric').rename(columns={'Value': name}))
    return y_pred_dict, pd.concat(results, axis=1)


def scale_predict(model, X_test: np.ndarray | pd.DataFrame, scaler):
    """
    Predict labels using the input model, optionally scaling the test data if a scaler is provided.

    :param model: The prediction model.
    :param X_test: Test features.
    :param scaler: Optional scaler object to transform the test data before prediction.
    :return: Predicted labels.
    """
    return model.predict(X_test if not scaler else scaler.transform(X_test))


def tune_hyperparameters(models_grid: dict, pipeline: Pipeline, scoring, X_train, y_train, save: str | None = None,
                         save_prefix: str = '',
                         model_step: str = 'model'):
    """
    Tune hyperparameters for a given set of models using GridSearchCV within a pipeline.

    :param models_grid: A dictionary containing models and their corresponding parameter grids.
    :param pipeline: The pipeline object.
    :param scoring: The scoring metrics.
    :param X_train: Training features.
    :param y_train: Training labels.
    :param save: Optional. Directory path to save the tuned models.
    :param save_prefix: Prefix to be added to the saved model filenames.
    :param model_step: The name of the step in the pipeline corresponding to the model.
    :return: A dictionary containing tuned models and relevant information about their performance
             as well as the scaler if provided in the pipeline.
    """
    start = time.time()
    tuned_models = {}
    for name, (model, param_grid) in models_grid.items():
        pipeline.steps.append((model_step, model))
        print(pipeline)
        print(f'Tuning hyperparameters for: {name} (time: {time.time() - start:.1f} s)')
        # an estimator will raise a warning in case you try to train it with an invalid combination of parameters
        # (i.e. LogisticRegression with solver='sag' and penalty='l1'), but it isn't further evaluated.
        # Therefore, the warning will be suppressed.
        param_grid = {model_step + '__' + key: val for key, val in param_grid.items()}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FitFailedWarning)
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            print(f'Hyperparameter values to test: {param_grid}')
            search_cv = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=scoring, refit='F1 Score', cv=4,
                                     n_jobs=-1, error_score=0, verbose=1)
            search_cv.fit(X_train, y_train)
        best_model = search_cv.best_estimator_.named_steps[model_step]
        scaler = search_cv.best_estimator_.named_steps.get('scaler', None)
        tuned_models[name] = {
            'model': best_model,
            'scaler': scaler,
            'f1_score_mean': search_cv.best_score_,
            'recall_mean': search_cv.cv_results_['mean_test_Recall'][search_cv.best_index_],
            'kappa_mean': search_cv.cv_results_['mean_test_Kappa Score'][search_cv.best_index_],
            'params': search_cv.best_params_
        }
        if save:
            save_data = (best_model, scaler)
            model_path = Path(save) / f'{save_prefix}{name}_tuned.pkl'
            joblib.dump(save_data, model_path)
            print(f'Tuned model saved at: {model_path}')
        pipeline.steps.pop()
    print(f'Hyperparameter tuning complete, took {time.time() - start:.2f} s')
    return tuned_models


def load_model(model_name):
    """
    Load a saved model along with its associated scaler.

    :param model_name: The file path of the saved model.
    :return: A tuple containing the loaded model and scaler.
    """
    model, scaler = joblib.load(model_name)
    return model, scaler
