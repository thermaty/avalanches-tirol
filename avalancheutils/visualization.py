import warnings
from typing import Iterable, Tuple

import folium
import matplotlib
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from folium import Icon, TileLayer
from folium.plugins import Fullscreen, HeatMap, MarkerCluster
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from rasterio.plot import show
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from . import config
import xyzservices.providers as xyz

def _check_ax(input_ax, figsize, dpi=200, subplot_kw=None):
    """
    Check if an Axes is provided and create one with given parameters if not.

    :param input_ax: Existing Axes or None.
    :param figsize: Figure size.
    :param dpi: Dots per inch (default: 200).
    :param subplot_kw: Keyword arguments for subplot creation (default: None).
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    if input_ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw=subplot_kw)
    else:
        fig, ax = input_ax.figure, input_ax
    ax.grid(axis='x', visible=False)
    return fig, ax


def plot_histogram(data, x_label: str, y_label: str, bins, ax: plt.Axes = None,
                   figsize: Tuple[int, int] = (8, 5), title: str = None, x_ticks_labels=None, y_ticks=None,
                   color='dodgerblue', label=None, rotation=0) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of the data.

    :param data: Data to be plotted.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param y_ticks: Optional. Ticks for y-axis.
    :param x_ticks_labels: Optional. Labels for x-axis ticks.
    :param rotation: Optional. Rotation angle for x-axis labels.
    :param label: Optional. Label for the legend.
    :param title: Optional. Title for the plot.
    :param bins: Number of bins or bin edges for the histogram.
    :param ax: Optional. Axes object to plot on.
    :param figsize: Optional. Figure size.
    :param color: Optional. Color for the histogram bars (default: 'dodgerblue').
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = _check_ax(ax, figsize)
    ax.hist(data, bins=bins, color=color, edgecolor='black', label=label, zorder=1)
    if title:
        ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    step_size = abs(bins[0] - bins[1])
    x_ticks_pos = [i + 0.5 * step_size for i in bins[:-1]]
    ax.set_xticks(ticks=x_ticks_pos, labels=x_ticks_labels, rotation=rotation)
    if y_ticks:
        ax.set_yticks(*y_ticks)
    return fig, ax


def plot_boxplot(data, title: str, x_label: str, y_label: str, figsize=(7, 3), x_ticks=None, ax=None):
    """
    Plot a boxplot of the data.

    :param data: Data to be plotted.
    :param title: Title for the plot.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    :param x_ticks: Optional. Ticks for x-axis.
    :param ax: Optional. Axes object to plot on.
    :param figsize: Optional. Figure size.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = _check_ax(ax, figsize)
    ax.boxplot(data, vert=False)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    if x_ticks:
        ax.set_xticks(x_ticks)
    ax.set_yticks([])
    ax.set_ylabel(y_label)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    return fig, ax


def plot_heatmap(matrix, title: str, figsize=(20, 20), trimask=False, color='vlag', center=None, ax=None):
    """
    Plot a heatmap of the input matrix.

    :param matrix: Matrix to plot.
    :param title: Title for the plot.
    :param trimask: Optional. Whether to trim the mask for a triangular heatmap.
    :param color: Optional. Colormap for the heatmap.
    :param center: Optional. Value to center the colormap around.
    :param ax: Optional. Axes object to plot on.
    :param figsize: Optional. Figure size.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = _check_ax(ax, figsize)
    mask = np.triu(np.ones_like(matrix, dtype=bool)) if trimask else None
    heatmap = sns.heatmap(matrix, annot=True, fmt=".2f", cmap=color, square=True, mask=mask, center=center, ax=ax,
                          cbar_kws={"shrink": .82})
    if trimask:
        y_ticks, y_labels = heatmap.get_yticks()[1:], heatmap.get_yticklabels()[1:]
        x_ticks, x_labels = heatmap.get_xticks()[:-1], heatmap.get_xticklabels()[:-1]
        heatmap.set_yticks(y_ticks, y_labels)
        heatmap.set_xticks(x_ticks, x_labels)
    ax.set_title(title)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.set_aspect('auto')
    return fig, ax


def plot_confusion_matrix(y_true, y_pred, target_name, figsize=(8, 8), ax=None, color='Blues', norm: str = None):
    """
    Plot the confusion matrix for predicted and true labels.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param target_name: Name of the target feature.
    :param figsize: Optional. Figure size.
    :param ax: Optional. Axes object to plot on (default: None).
    :param color: Optional. Colormap for the plot (default: 'Blues').
    :param norm: Optional. Sklearn's normalization mode for the confusion matrix.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = _check_ax(ax, figsize)
    ax.grid(False)
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, normalize=norm)).plot(ax=ax, cmap=color)
    ax.set_title(f'Confusion Matrix for the target feature {target_name}')
    return fig, ax


def plot_pie_chart(data: pd.Series, title: str = None, figsize=(6, 5), labels=None,
                   startangle=90, ax=None):
    """
    Plot a pie chart.

    :param data: Data to be plotted.
    :param title: Optional. Title for the plot.
    :param figsize: Optional. Figure size (default: (6, 5)).
    :param labels: Optional. Labels for the pie chart slices.
    :param startangle: Optional. Start angle for the first slice (default: 90).
    :param ax: Optional. Axes object to plot on.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    fig, ax = _check_ax(ax, figsize)
    ax.pie(data, labels=labels,
           autopct=lambda x: f'{x:.02f}% ({x * data.sum() / 100:.0f})',
           startangle=startangle)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_polar_bar(data: dict, title: str = None, x_ticks: Iterable[str] = None, color='dodgerblue', norm_color=False,
                   y_ticks: Iterable = None, figsize=(8, 8), ax=None, label=None):
    """
    Plot a polar bar chart. For ordered and uniformly distributed circular data.

    :param data: Dictionary containing the data to plot.
    :param title: Optional. Title for the plot.
    :param x_ticks: Optional. Ticks for the x-axis.
    :param color: Optional. Color for the bars.
    :param norm_color: Optional. Whether to normalize the bar colors.
    :param y_ticks: Optional. Ticks for the y-axis.
    :param ax: Optional. Axes object to plot on.
    :param figsize: Optional. Figure size.
    :param label: Optional. Label for the legend.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    angles = np.linspace(0, 2 * np.pi, len(data), endpoint=False)
    fig, ax = _check_ax(ax, figsize=figsize, subplot_kw={'projection': 'polar'})
    bars = ax.bar(angles, data.values(), width=np.pi / 4 - 0.05, align='center', color=color, label=label)
    if norm_color:
        norm = Normalize(vmin=min(data.values()), vmax=max(data.values()))
        color_map = matplotlib.colormaps[norm_color]
        for bar, count in zip(bars, data.values()):
            color = color_map(norm(count))
            bar.set_facecolor(color)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    if x_ticks:
        ax.set_xticks(ticks=angles, labels=x_ticks)
    if y_ticks:
        ax.set_yticks(*y_ticks)
    ax.set_rlabel_position(90)
    ax.yaxis.set_label_coords(0, 0.1)
    if title:
        ax.set_title(title)
    return fig, ax


def plot_dem(dem_file, save=None, cmap='terrain', dpi: int = 300, colorbar: str = None, transparent=True,
             title='') -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot digital elevation model (DEM) data.

    :param dem_file: File path to the DEM data.
    :param save: Optional. File path to save the plot.
    :param cmap: Optional. Colormap for the plot.
    :param dpi: Optional. Dots per inch for the plot resolution.
    :param colorbar: Optional. Label for the colorbar. If None, colorbar will not be included.
    :param transparent: Optional. Whether to make the plot background transparent.
    :param title: Optional. Title for the plot.
    :return: Tuple containing the matplotlib Figure and Axes objects.
    """
    dem_data = rasterio.open(dem_file)
    fig, ax = plt.subplots(figsize=(13, 6))
    plotted = show(dem_data, ax=ax, cmap=cmap, aspect='equal')
    im = plotted.get_images()[0]
    ax.set_xticks([]), ax.set_yticks([]), ax.set_xlabel(''), ax.set_ylabel(''), ax.set_title(title)
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
        cbar.set_label(colorbar)
        transparent = False
    if save:
        plt.savefig(save, dpi=dpi, bbox_inches='tight', pad_inches=0.2, transparent=transparent)
    return fig, ax


def plot_incident_map(data: pd.DataFrame, lon_idx, lat_idx, popup_data: bool = False, heatmap: bool = False,
                      zoom: int = 4, additional_tiles=None) -> folium.Map:
    """
    Plot incidents on a folium map.

    :param data: DataFrame containing incidents.
    :param lon_idx: Name or index of the column containing longitude data.
    :param lat_idx: Name or index of the column containing latitude data.
    :param popup_data: Optional. Whether to include popup data.
    :param heatmap: Optional. Whether to include a heatmap.
    :param zoom: Optional. Zoom level for the map.
    :param additional_tiles: Optional. Additional tile layers.
    :return: folium.Map object containing the records from data.
    """
    if additional_tiles is None:
        additional_tiles = []
    center = (np.mean(data[lat_idx]), np.mean(data[lon_idx]))
    result_map = folium.Map(location=center, zoom_start=zoom)
    _map_add_custom_tiles(result_map)
    for additional_tiles in additional_tiles:
        TileLayer(tiles=additional_tiles, show=False).add_to(result_map)
    marker_cluster = MarkerCluster()
    incident_group = folium.FeatureGroup(name='Incidents')
    _map_add_fullscreen(result_map)
    for _, incident in data.iterrows():
        incident_loc = (incident[lat_idx], incident[lon_idx])
        popup = _create_table_html(incident, lon_idx, lat_idx) if popup_data else None
        folium.Marker(location=incident_loc, icon=Icon(prefix='fa', icon='hill-avalanche'), popup=popup).add_to(
            marker_cluster).add_to(marker_cluster)
    marker_cluster.add_to(incident_group)
    incident_group.add_to(result_map)
    if heatmap:
        _map_add_heatmap(data, lon_idx, lat_idx, result_map)
    folium.LayerControl(collapsed=True).add_to(result_map)
    return result_map


def _map_add_custom_tiles(m: folium.Map):
    """
    Adds custom tiles to a Folium map based on configurations from a config file.

    :param m: Folium Map object to which the custom tiles will be added.
    """
    config_dict = config.load_config(config_path='config.yml', key='folium_maps')
    if not (tiles_name := config_dict.get('xyz_tiles_name')):
        return
    tiles = xyz.flatten().get(tiles_name)
    if not tiles:
        warnings.warn('Could not get the tiles provider, please check the name in the config file.')
        return
    if 'apikey' in tiles:
        api_key = config_dict.get('tiles_api_key')
        if not api_key:
            warnings.warn('Invalid API key.')
            return
        tiles['apikey'] = api_key
    TileLayer(tiles).add_to(m)


def _map_add_fullscreen(m: folium.Map):
    """
    Add fullscreen control to the folium map.

    :param m: folium.Map object to add the fullscreen control to.
    """
    Fullscreen(
        position='topright',
        force_separate_button=True,
    ).add_to(m)


def _map_add_heatmap(data: pd.DataFrame, lon_idx, lat_idx, result_map: folium.Map):
    """
    Add heatmap layer to the folium map.

    :param data: DataFrame containing incident data.
    :param lon_idx: Name or index of the column containing longitude data.
    :param lat_idx: Name or index of the column containing latitude data.
    :param result_map: folium.Map object to add the heatmap layer to.
    """
    HeatMap(data[[lat_idx, lon_idx]].values, min_opacity=0.2, radius=16, blur=5, name='Heatmap').add_to(result_map)


def _create_table_html(incident_data, lon_idx, lat_idx) -> str:
    """
    Create HTML table for incident map popup menu.

    :param incident_data: Data of the incident.
    :param lon_idx: Name or index of the column containing longitude data.
    :param lat_idx: Name or index of the column containing latitude data.
    :return: HTML string representing the incident popup menu.
    """
    table_html = '<b>Incident Details</b>'
    table_html += f'<p>Latitude: {incident_data.pop(lat_idx)}</p>'
    table_html += f'<p>Longitude: {incident_data.pop(lon_idx)}</p>'
    if incident_data.empty is False:
        table_html += '<table style="border-collapse: collapse; border: 1px solid black;">'
        table_html += '<style>td { border: 1px solid black; padding: 5px; white-space: nowrap; }</style>'

        for key, value in incident_data.items():
            table_html += f'<tr><td>{key}</td><td>{value}</td></tr>'
        table_html += '</table>'
    return table_html
