# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 23:37:00 2023

@author: velan
"""

# 2000


import rasterio
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from rasterio.plot import show
from matplotlib.pyplot import imread


#code for Temperature Condition Index
file_path = "C:/HARSH/EOS_project/MAIN/subset_img/subset_2000/subset2000.img"

with rasterio.open(file_path) as dataset:
    # Create an empty DataFrame
    dfr = pd.DataFrame()

    # Loop through each band
    for band_index in range(1, dataset.count + 1):
        # Read the band data
        band_data = dataset.read(band_index)

        # Flatten the 2D band array to a 1D array
        flattened_band = band_data.flatten()

        # Create a column name for the band
        column_name = f"Band_{band_index}"

        # Add the band data to the DataFrame
        dfr[column_name] = flattened_band

# Display the DataFrame
print(dfr.head())
print(dfr["Band_1"])
dfr.shape


#applying z score normalization
sc = StandardScaler()
val = dfr['Band_1'].values
val
dfr['Band_1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = dfr['Band_2'].values
dfr['Band_2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_3'].values
dfr['Band_3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_4'].values
dfr['Band_4'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_5'].values
dfr['Band_5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_6'].values
dfr['Band_6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_7'].values
dfr['Band_7'] = sc.fit_transform(val.reshape(-1, 1))




df = pd.read_csv('C:/HARSH/EOS_project/MAIN/csv/point_raster_2022.csv')
# for 1st 5 rows
df.head()

df.columns
# all column names

# rows and columns(features)
df.shape

# to identify any null values
df.isnull().sum()

# Statistics of numerical columns only
df.describe()

le = preprocessing.LabelEncoder()


label_le = preprocessing.LabelEncoder()
label_le.fit(list(df['Name']))
df['label_cat'] = df['Name'].apply(lambda x: label_le.transform([x])[0])
df[['Name', 'label_cat']]

data = df.copy()

data.head()

data.columns

data_df = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
data_df.head()

#new_data = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
#sns.heatmap(data=new_data.corr(), lw = 1)

fdata = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','label_cat']]
fdata

#applying z score normalization
sc = StandardScaler()
val = fdata['B1'].values
fdata['B1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = fdata['B2'].values
fdata['B2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B3'].values
fdata['B3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B4'].values
fdata['B4'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B5'].values
fdata['B5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B6'].values
fdata['B6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B7'].values
fdata['B7'] = sc.fit_transform(val.reshape(-1, 1))

fdata.head()



'''here test data will be trained by using mean, variance training data'''



X = fdata.drop('label_cat', axis = 1).values     #dropping the last column i.e lables from data
y = fdata['label_cat'].values   #making the target vector

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
# 0.25 means 25 percent data for test

#y_test = dfr
print(dfr.shape)
size_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = size_scaler.transform(X_train)
X_test_scaled = size_scaler.transform(X_test)
X_train_scaled.shape, X_test_scaled.shape

rf = RandomForestClassifier(max_depth=4)   # loading the random forest classifier
rf.fit(X_train, y_train)       # Training the random forest classifier
rf_yhat = rf.predict(X_test)   # Predicting on the test dataset

print('Accuracy score of the Random Forest model is ',accuracy_score(y_test, rf_yhat))

print('F1 Score of the Random Forest model is ',f1_score(y_test, rf_yhat, average='macro'))

cm = confusion_matrix(y_test, rf_yhat, labels = [0, 1, 2])
print(cm)

y_new=rf.predict(dfr)
print(y_new.shape)
y_new=y_new.reshape(1,-1)
y_new=np.transpose(y_new)
print(y_new.shape)


dfr=np.hstack((dfr,y_new))
dfr

symbology2 =  { 19: 'vegetation',
              5: 'snow',
            13: 'barren'}

# Visualize
# Because the predicted labels are still in one column, you need to reshape it back to original image shape

row, col = dataset.shape  # Get the original dimensions of the image
imin = min(symbology2)  # Colormap range
imax = max(symbology2)
print("Minimum Values : ",imin )
print("Maximum Values : ", imax)



# Again we use our image data for georeferencing information
rst = rasterio.open(file_path)
meta = rst.meta.copy()  # Copy metadata from the base image
meta.update(compress='lzw')

# Burn the AOIs *.shp file into raster and save it
out_rst = 'rf_prediction_2000.tif'
out_file = rasterio.open(
    out_rst,
    'w',
    driver='GTiff',
    height=row,
    width=col,
    count=1,
    dtype=y_new.dtype,
    crs=rst.crs,
    transform=rst.transform)

out_file.write(y_new.reshape(row, col),1)
out_file.close()
rf_tif = rasterio.open('rf_prediction_2000.tif')
show(rf_tif)


unique, counts = np.unique(y_new, return_counts=True)
results = np.column_stack((unique,counts))
summed_counts = sum(counts)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classes = ['1 =VEGETATION', '2 = SNOW ','3= BARREN']
ax.set_title('Percentages of the Predictions of Each Class for Dec-2000')
ax.set_xlabel('Classes')
ax.set_ylabel('Percentage')
ax.bar(classes, counts/summed_counts)
plt.show()



# Area Calculations
l_0_2000 = 0
l_1_2000 = 0
l_2_2000 = 0
for i in y_new:
    for j in i:
        if j == 0:
            l_0_2000 += 1
        elif j == 1:
            l_1_2000 += 1
        elif j == 2:
            l_2_2000 += 1
veg_area_2000 = (l_0_2000*30*30)/10**6
snow_area_2000 = (l_1_2000*30*30)/10**6
brn_area_2000 = (l_2_2000*30*30)/10**6
print(f'Vegetation Area : {veg_area_2000} sq. km')
print(f'Snow Area : {snow_area_2000} sq. km')
print(f'Barren Area : {brn_area_2000} sq. km')



print("******************************************************************************")


# 2010

#code for Temperature Condition Index
file_path = "C:/HARSH/EOS_project/MAIN/subset_img/subset_2010/subset2010.img"

with rasterio.open(file_path) as dataset:
    # Create an empty DataFrame
    dfr = pd.DataFrame()

    # Loop through each band
    for band_index in range(1, dataset.count + 1):
        # Read the band data
        band_data = dataset.read(band_index)

        # Flatten the 2D band array to a 1D array
        flattened_band = band_data.flatten()

        # Create a column name for the band
        column_name = f"Band_{band_index}"

        # Add the band data to the DataFrame
        dfr[column_name] = flattened_band

# Display the DataFrame
print(dfr.head())
print(dfr["Band_1"])


#applying z score normalization
sc = StandardScaler()
val = dfr['Band_1'].values
val
dfr['Band_1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = dfr['Band_2'].values
dfr['Band_2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_3'].values
dfr['Band_3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_4'].values
dfr['Band_4'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_5'].values
dfr['Band_5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_6'].values
dfr['Band_6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_7'].values
dfr['Band_7'] = sc.fit_transform(val.reshape(-1, 1))




df = pd.read_csv('C:/HARSH/EOS_project/MAIN/csv/point_raster_2022.csv')
# for 1st 5 rows
df.head()

df.columns
# all column names

# rows and columns(features)
df.shape

# to identify any null values
df.isnull().sum()

# Statistics of numerical columns only
df.describe()

le = preprocessing.LabelEncoder()


label_le = preprocessing.LabelEncoder()
label_le.fit(list(df['Name']))
df['label_cat'] = df['Name'].apply(lambda x: label_le.transform([x])[0])
df[['Name', 'label_cat']]

data = df.copy()

data.head()

data.columns

data_df = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
data_df.head()

#new_data = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
#sns.heatmap(data=new_data.corr(), lw = 1)

fdata = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','label_cat']]
fdata

#applying z score normalization
sc = StandardScaler()
val = fdata['B1'].values
fdata['B1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = fdata['B2'].values
fdata['B2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B3'].values
fdata['B3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B4'].values
fdata['B4'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B5'].values
fdata['B5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B6'].values
fdata['B6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B7'].values
fdata['B7'] = sc.fit_transform(val.reshape(-1, 1))

fdata.head()



'''here test data will be trained by using mean, variance training data'''



X = fdata.drop('label_cat', axis = 1).values     #dropping the last column i.e lables from data
y = fdata['label_cat'].values   #making the target vector

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
# 0.25 means 25 percent data for test

#y_test = dfr
print(dfr.shape)
size_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = size_scaler.transform(X_train)
X_test_scaled = size_scaler.transform(X_test)
X_train_scaled.shape, X_test_scaled.shape

rf = RandomForestClassifier(max_depth=4)   # loading the random forest classifier
rf.fit(X_train, y_train)       # Training the random forest classifier
rf_yhat = rf.predict(X_test)   # Predicting on the test dataset

print('Accuracy score of the Random Forest model is ',accuracy_score(y_test, rf_yhat))

print('F1 Score of the Random Forest model is ',f1_score(y_test, rf_yhat, average='macro'))

cm = confusion_matrix(y_test, rf_yhat, labels = [0, 1, 2])
print(cm)

y_new=rf.predict(dfr)
print(y_new.shape)
y_new=y_new.reshape(1,-1)
y_new=np.transpose(y_new)
print(y_new.shape)


dfr=np.hstack((dfr,y_new))
dfr

symbology2 =  { 19: 'vegetation',
              5: 'snow',
            13: 'barren'}


row, col = dataset.shape  # Get the original dimensions of the image
imin = min(symbology2)  # Colormap range
imax = max(symbology2)
print("Minimum Values : ",imin )
print("Maximum Values : ", imax)



# Again we use our image data for georeferencing information
rst = rasterio.open(file_path)
meta = rst.meta.copy()  # Copy metadata from the base image
meta.update(compress='lzw')

# Burn the AOIs *.shp file into raster and save it
out_rst = 'rf_prediction_2010.tif'
out_file = rasterio.open(
    out_rst,
    'w',
    driver='GTiff',
    height=row,
    width=col,
    count=1,
    dtype=y_new.dtype,
    crs=rst.crs,
    transform=rst.transform)

out_file.write(y_new.reshape(row, col),1)
out_file.close()
rf_tif = rasterio.open('rf_prediction_2010.tif')
show(rf_tif)


unique, counts = np.unique(y_new, return_counts=True)
results = np.column_stack((unique,counts))
summed_counts = sum(counts)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classes = ['1 =VEGETATION', '2 = SNOW ','3= BARREN']
ax.set_title('Percentages of the Predictions of Each Class for Dec-2010')
ax.set_xlabel('Classes')
ax.set_ylabel('Percentage')
ax.bar(classes, counts/summed_counts)
plt.show()



# Area Calculations
l_0_2010 = 0
l_1_2010 = 0
l_2_2010 = 0
for i in y_new:
    for j in i:
        if j == 0:
            l_0_2010 += 1
        elif j == 1:
            l_1_2010 += 1
        elif j == 2:
            l_2_2010 += 1
veg_area_2010 = (l_0_2010*30*30)/10**6
snow_area_2010 = (l_1_2010*30*30)/10**6
brn_area_2010 = (l_2_2010*30*30)/10**6
print(f'Vegetation Area : {veg_area_2010} sq. km')
print(f'Snow Area : {snow_area_2010} sq. km')
print(f'Barren Area : {brn_area_2010} sq. km')




print("******************************************************************************")





# 2022

#code for Temperature Condition Index
file_path = "C:/HARSH/EOS_project/MAIN/subset_img/subset_2022/subset2022.img"

with rasterio.open(file_path) as dataset:
    # Create an empty DataFrame
    dfr = pd.DataFrame()

    # Loop through each band
    for band_index in range(1, dataset.count + 1):
        # Read the band data
        band_data = dataset.read(band_index)

        # Flatten the 2D band array to a 1D array
        flattened_band = band_data.flatten()

        # Create a column name for the band
        column_name = f"Band_{band_index}"

        # Add the band data to the DataFrame
        dfr[column_name] = flattened_band

# Display the DataFrame
print(dfr.head())
print(dfr["Band_1"])
dfr.shape


#applying z score normalization
sc = StandardScaler()
val = dfr['Band_1'].values
val
dfr['Band_1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = dfr['Band_2'].values
dfr['Band_2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_3'].values
dfr['Band_3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_4'].values
dfr['Band_4'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_5'].values
dfr['Band_5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_6'].values
dfr['Band_6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = dfr['Band_7'].values
dfr['Band_7'] = sc.fit_transform(val.reshape(-1, 1))




df = pd.read_csv('C:/HARSH/EOS_project/MAIN/csv/point_raster_2022.csv')
# for 1st 5 rows
df.head()

df.columns
# all column names

# rows and columns(features)
df.shape

# to identify any null values
df.isnull().sum()

# Statistics of numerical columns only
df.describe()

le = preprocessing.LabelEncoder()


label_le = preprocessing.LabelEncoder()
label_le.fit(list(df['Name']))
df['label_cat'] = df['Name'].apply(lambda x: label_le.transform([x])[0])
df[['Name', 'label_cat']]

data = df.copy()

data.head()

data.columns

data_df = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
data_df.head()

#new_data = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'label_cat']]
#sns.heatmap(data=new_data.corr(), lw = 1)

fdata = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7','label_cat']]
fdata
 
#applying z score normalization
sc = StandardScaler()
val = fdata['B1'].values
fdata['B1'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()  #applying z score normalization
val = fdata['B2'].values
fdata['B2'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B3'].values
fdata['B3'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B4'].values
fdata['B4'] = sc.fit_transform(val.reshape(-1, 1))


sc = StandardScaler()                          #applying z score normalization
val = fdata['B5'].values
fdata['B5'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B6'].values
fdata['B6'] = sc.fit_transform(val.reshape(-1, 1))

sc = StandardScaler()                          #applying z score normalization
val = fdata['B7'].values
fdata['B7'] = sc.fit_transform(val.reshape(-1, 1))

fdata.head()


'''here test data will be trained by using mean, variance training data'''



X = fdata.drop('label_cat', axis = 1).values     #dropping the last column i.e lables from data
y = fdata['label_cat'].values   #making the target vector

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.25)
# 0.25 means 25 percent data for test

#y_test = dfr
print(dfr.shape)
size_scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = size_scaler.transform(X_train)
X_test_scaled = size_scaler.transform(X_test)
X_train_scaled.shape, X_test_scaled.shape

rf = RandomForestClassifier(max_depth=4)   # loading the random forest classifier
rf.fit(X_train, y_train)       # Training the random forest classifier
rf_yhat = rf.predict(X_test)   # Predicting on the test dataset

print('Accuracy score of the Random Forest model is ',accuracy_score(y_test, rf_yhat))

print('F1 Score of the Random Forest model is ',f1_score(y_test, rf_yhat, average='macro'))

cm = confusion_matrix(y_test, rf_yhat, labels = [0, 1, 2])
print(cm)

y_new=rf.predict(dfr)
print(y_new.shape)
y_new=y_new.reshape(1,-1)
y_new=np.transpose(y_new)
print(y_new.shape)




dfr=np.hstack((dfr,y_new))
dfr

symbology2 =  { 19: 'vegetation',
              5: 'snow',
            13: 'barren'}

# Visualize
# Because the predicted labels are still in one column, you need to reshape it back to original image shape

row, col = dataset.shape  # Get the original dimensions of the image
imin = min(symbology2)  # Colormap range
imax = max(symbology2)
print("Minimum Values : ", imin )
print("Maximum Values : ", imax)



# Again we use our image data for georeferencing information
rst = rasterio.open(file_path)
meta = rst.meta.copy()  # Copy metadata from the base image
meta.update(compress='lzw')

# Burn the AOIs *.shp file into raster and save it
out_rst = 'rf_prediction_2022.tif'
out_file = rasterio.open(
    out_rst,
    'w',
    driver='GTiff',
    height=row,
    width=col,
    count=1,
    dtype=y_new.dtype,
    crs=rst.crs,
    transform=rst.transform)

out_file.write(y_new.reshape(row, col),1)
out_file.close()
rf_tif = rasterio.open('rf_prediction_2022.tif')
show(rf_tif)


unique, counts = np.unique(y_new, return_counts=True)
results = np.column_stack((unique,counts))
summed_counts = sum(counts)

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
classes = ['1 =VEGETATION', '2 = SNOW ','3= BARREN']
ax.set_title('Percentages of the Predictions of Each Class for Dec-2022')
ax.set_xlabel('Classes')
ax.set_ylabel('Percentage')
ax.bar(classes, counts/summed_counts)
plt.show()

# Area Calculations
l_0_2022 = 0
l_1_2022 = 0
l_2_2022 = 0
for i in y_new:
    for j in i:
        if j == 0:
            l_0_2022 += 1
        elif j == 1:
            l_1_2022 += 1
        elif j == 2:
            l_2_2022 += 1
veg_area_2022 = (l_0_2022*30*30)/10**6
snow_area_2022 = (l_1_2022*30*30)/10**6
brn_area_2022 = (l_2_2022*30*30)/10**6
print(f'Vegetation Area : {veg_area_2022} sq. km')
print(f'Snow Area : {snow_area_2022} sq. km')
print(f'Barren Area : {brn_area_2022} sq. km')


print("******************************************************************************")


veg_ch_2000_2010 = veg_area_2000 - veg_area_2010
print(f'Change Of Vegetation Area From 2000 to 2010 is {veg_ch_2000_2010} sq. km')

veg_ch_2010_2022 = veg_area_2010 - veg_area_2022
print(f'Change Of Vegetation Area From 2010 to 2022 is {veg_ch_2010_2022} sq. km')

veg_ch_2000_2022 = veg_area_2000 - veg_area_2022
print(f'Change Of Vegetation Area From 2000 to 2022 is {veg_ch_2000_2022} sq. km')
print("******************************************************************************")

snow_ch_2000_2010 = snow_area_2000 - snow_area_2010
print(f'Change Of Snow Area From 2000 to 2010 is {snow_ch_2000_2010} sq. km')

snow_ch_2010_2022 = snow_area_2010 - snow_area_2022
print(f'Change Of Snow Area From 2010 to 2022 is {snow_ch_2010_2022} sq. km')

snow_ch_2000_2022 = snow_area_2000 - snow_area_2022
print(f'Change Of Snow Area From 2000 to 2022 is {snow_ch_2000_2022} sq. km')
print("******************************************************************************")

brn_ch_2000_2010 = brn_area_2000 - brn_area_2010
print(f'Change Of Barren Area From 2000 to 2010 is {brn_ch_2000_2010} sq. km')

brn_ch_2010_2022 = brn_area_2010 - brn_area_2022
print(f'Change Of Barren Area From 2010 to 2022 is {brn_ch_2010_2022} sq. km')

brn_ch_2000_2022 = brn_area_2000 - brn_area_2022
print(f'Change Of Barren Area From 2000 to 2022 is {brn_ch_2000_2022} sq. km')