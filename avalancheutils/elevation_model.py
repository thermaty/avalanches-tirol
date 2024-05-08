import os
from functools import cache
from pathlib import Path
from typing import Tuple

import pyproj
from osgeo import gdal

# by default gdal is not throwing exceptions in case of an error
gdal.UseExceptions()


class ElevationModel:
    """
    A wrapper class for manipulating terrain data with GDAL.

    This class provides methods to access the elevation, aspect and slope data calculated from the elevation model.
    """
    def __init__(self, elevation_file: str, aspect_file: str = None, slope_file: str = None):
        """
        Initialize the ElevationModel object.

        :param elevation_file: Path to the elevation model file.
        :param aspect_file: Optional. Path to the aspect model file.
        :param slope_file: Optional. Path to the slope model file.
        """
        self.ds = gdal.Open(elevation_file)
        self.slope_file = slope_file
        self.aspect_file = aspect_file
        self._calculate_aspect()
        self._calculate_slope()
        self.elevation = self.ds
        self.aspect = self._load_model(self.aspect_file)
        self.slope = self._load_model(self.slope_file)
        self.transform = self.ds.GetGeoTransform()

    def get_elevation(self, lon: float, lat: float):
        """
        Get the elevation at a specified longitude and latitude WGS84 coordinates.

        :param lon: Longitude value.
        :param lat: Latitude value.
        :return: Elevation in meters.
        """
        p_x, p_y = self._process_coords(lon, lat)
        return self.elevation.GetRasterBand(1).ReadAsArray(p_x, p_y, 1, 1)[0][0]

    def get_aspect(self, lon: float, lat: float):
        """
        Get the slope aspect at a specified longitude and latitude WGS84 coordinates.

        :param lon: Longitude value.
        :param lat: Latitude value.
        :return: Aspect value.
        """
        p_x, p_y = self._process_coords(lon, lat)
        return self.aspect.GetRasterBand(1).ReadAsArray(p_x, p_y, 1, 1)[0][0]

    def get_slope(self, lon: float, lat: float):
        p_x, p_y = self._process_coords(lon, lat)
        return self.slope.GetRasterBand(1).ReadAsArray(p_x, p_y, 1, 1)[0][0]

    def _calculate_aspect(self):
        """
        Calculate slope aspect model if a valid aspect model file is not provided.
        """
        if self.aspect_file is None:
            self.aspect_file = 'aspect.tif'
        if not Path(self.aspect_file).exists():
            print('Creating the aspect model.')
            gdal.DEMProcessing(self.aspect_file, self.ds, 'aspect', computeEdges=True, trigonometric=False)
            print(f'Aspect model successfully saved to file {self.aspect_file}')

    def _calculate_slope(self):
        """
        Calculate slope angle model if a valid slope angle model file is not provided.
        """
        if self.slope_file is None:
            self.slope_file = 'slope.tif'
        if not Path(self.slope_file).exists():
            print('Creating the slope model.')
            gdal.DEMProcessing(self.slope_file, self.ds, 'slope', computeEdges=True, slopeFormat='degree')
            print(f'Slope model successfully saved to file {self.slope_file}')

    @staticmethod
    def _load_model(model_file: str, remove_output=False):
        """
        Load a GIS model file with GDAL.

        :param model_file: Path to the model file.
        :param remove_output: Whether to remove the output file after loading.
        :return: Loaded model dataset.
        """
        model_ds = gdal.Open(model_file)
        if remove_output:
            os.remove(model_file)
        return model_ds

    @cache
    def _process_coords(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Process coordinates to convert them to pixel coordinates of the raster model. First convert them to the EPSG
        format of the model and then use affine projection to map them to pixel coordinates in the raster.

        :param lon: Longitude value.
        :param lat: Latitude value.
        :return: Pixel coordinates tuple.
        """
        converted_coords = ElevationModel.convert_coords_formats(lon, lat)
        return self._coords_to_pixels(*converted_coords)

    def _coords_to_pixels(self, lon: float, lat: float) -> Tuple[int, int]:
        """
        Convert coordinates to pixel coordinates.

        :param lon: Longitude value.
        :param lat: Latitude value.
        :return: Pixel coordinates tuple.
        """
        p_x = int((lon - self.transform[0]) / self.transform[1])
        p_y = int((lat - self.transform[3]) / self.transform[5])
        return p_x, p_y

    @staticmethod
    def convert_coords_formats(lon: float, lat: float, source='EPSG:4326', target='EPSG:31254'):
        """
        Convert coordinates from one format to another.

        :param lon: Longitude value.
        :param lat: Latitude value.
        :param source: Source coordinate system (default: 'EPSG:4326' - WGS84).
        :param target: Target coordinate system (default: 'EPSG:31254').
        :return: Converted coordinates as a tuple.
        """
        transformer = pyproj.Transformer.from_crs(source, target, always_xy=True)
        return transformer.transform(lon, lat)
