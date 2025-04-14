# EOS Project: Remote Sensing Analysis 

This project performs remote sensing analysis using satellite imagery data from different years (2000, 2010, 2022). It leverages machine learning techniques, particularly the Random Forest Classifier, to classify land cover types such as vegetation, snow, and barren land. The project also includes geospatial data analysis and area calculations for each land class.

## Project Structure

The project is structured as follows:

- **subset_img/**: Contains the satellite imagery files from 2000, 2010, and 2022 in `.img` format.
- **csv/**: Contains point data (raster data points) for training the model, especially for the year 2022.
- **scripts/**: Contains the Python scripts for processing, analyzing, and classifying the satellite imagery using machine learning models.

## Key Features

- **Data Preprocessing**: The images from each year (2000, 2010, 2022) are read, and band data is extracted using the `rasterio` library.
- **Z-Score Normalization**: Band data from the satellite images is normalized using the `StandardScaler` for better training of the machine learning models.
- **Model Training and Testing**: Random Forest Classifier is used to train on labeled data and test predictions on new data. The model is trained with features from 7 bands of satellite images.
- **Land Cover Classification**: The classification labels for each pixel in the image are assigned to vegetation, snow, and barren land categories based on predictions made by the model.
- **Geospatial Output**: The classified output is saved as a GeoTIFF file (`rf_prediction_2000.tif`, `rf_prediction_2010.tif`, `rf_prediction_2022.tif`), which can be visualized on mapping platforms.
- **Area Calculation**: The total area covered by each land class is computed and output in square kilometers.

## Libraries and Dependencies

This project requires the following Python libraries:

- `rasterio`
- `numpy`
- `pandas`
- `sklearn`
- `matplotlib`
- `seaborn`
- `scipy`

You can install the required dependencies using `pip`

## Usage

### Data Preparation

1. **Download the Satellite Imagery**:
   - Download the satellite imagery data for the years 2000, 2010, and 2022.
   - Place the images in the appropriate directories:
     - `subset_img/subset_2000/`
     - `subset_img/subset_2010/`
     - `subset_img/subset_2022/`

2. **Prepare the CSV Data**:
   - Prepare the CSV data (containing labeled data points) and place it in the `csv/` directory.

### Run the Scripts

The project contains Python scripts for each year. You can run them individually based on your needs:

- `EOS_ML_2000.py`: Runs the model for the year 2000.
- `EOS_ML_2010.py`: Runs the model for the year 2010.
- `EOS_ML_2022.py`: Runs the model for the year 2022.

### Output

Each script generates the following output:

- A classified image in **GeoTIFF** format showing the land cover classification for the respective year.
- A bar plot of the classification percentages for vegetation, snow, and barren land is also displayed.

### Area Calculation

- The scripts calculate and print the total area covered by each land class (vegetation, snow, barren) in **square kilometers**.

# Results and Visualization
The output GeoTIFF files can be visualized using GIS software like QGIS or ArcGIS. Additionally, the classification results and area calculations are displayed in the terminal or command prompt.
