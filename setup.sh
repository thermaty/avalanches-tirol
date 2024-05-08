#!/bin/bash

DEM_URL="https://gis.tirol.gv.at/ogd/geografie_planung/DGM/DGM_Tirol_5m_epsg31254.zip"
DEST_FOLDER="assets/geo_models"
TIFF_FILENAME="elevation_tirol.tif"

SKIP_DEM=false
while getopts "n" opt; do
  case $opt in
    n)
      SKIP_DEM=true
      ;;
    *)
      echo "Usage: $0 [-n]"
      exit 1
      ;;
  esac
done

# Prepare the python virtual environment
if [ ! -d ".venv" ]; then
   python3 -m venv .venv
fi
source .venv/bin/activate
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Failed to activate the virtual environment"
    exit 1
else
    echo "Virtual environment activated"
fi
pip3 install --no-cache-dir -r requirements.txt
pip3 install --no-build-isolation --no-cache-dir --force-reinstall GDAL=="$(gdal-config --version)"




if [ "$SKIP_DEM" = false ]; then
   mkdir -p "$DEST_FOLDER"
   echo "Downloading the DEM model."
   # Check if the download was successful
   if curl -o dem_tirol.zip "$DEM_URL"; then
       # Unzip the file into the specified folder
       if ! unzip -o dem_tirol.zip "*.tif" -d "$DEST_FOLDER"; then
           echo "Couldn't extract the zip file."
           exit 1
       fi
       mv "$DEST_FOLDER"/*.tif "$DEST_FOLDER/$TIFF_FILENAME"
       echo "The DEM file has been successfully extracted and renamed to $TIFF_FILENAME."
       rm dem_tirol.zip
   else
       echo "Download failed."
       exit 1
   fi
else
   echo "Skipping DEM processing due to -n flag."
fi
