import numpy as np 
import math
# dataset credit: https://www.kaggle.com/datasets/tushardobal/india-rainfall-and-crop-yield-dataset-20142024/data
#written by ashutosh sononey

#data = np.genfromtxt('india_rainfall_crop_yield_2014_2024.csv',delimiter=',',skip_header=1,
#                     dtype=[('record','i4'),('col2','S100'),
#                            ('col3','S100'),
#                            ('col4','i4'),('col5','S100'),
#                            ('col6','S100'),('col7','f8'),
#                            ('col8','f8'),('col9','f8'),
#                            ('col10','S100'),('col11','S100'),
#                            ('col12','f8'),('col13','f8'),
#                            ('col14','f8'),('col15','f8'),
#                            ('col16','f8'),('col17','f8'),]
#                     )

# print(data)


data = np.genfromtxt('india_rainfall_crop_yield_2014_2024.csv',delimiter=',',skip_header=1)
# print([data[:math.floor(len(data)*0.8)],data[math.floor(len(data)*0.8):]])

test =data[:math.floor(len(data)*0.8)]
train = data[math.floor(len(data)*0.8):]
print("Initial size:",len(data))
print("split 80/20",len(test),len(train),"total:",len(test)+len(train))

test_rainfall_mm=test[::,6]
test_crop_yield_hectare=test[::,14]
train_rainfall_mm=train[::,6]
train_crop_yield_hectare=train[::,14]

print("Test mean: rainfall (in mm) v/s cropy yield (in hectare)",np.mean(test_rainfall_mm),np.mean(test_crop_yield_hectare))
print("Train mean: rainfall (in mm) v/s cropy yield (in hectare)",np.mean(train_rainfall_mm),np.mean(train_crop_yield_hectare))
print("Test var: rainfall (in mm) v/s cropy yield (in hectare)",np.var(test_rainfall_mm),np.var(test_crop_yield_hectare))
print("Train var: rainfall (in mm) v/s cropy yield (in hectare)",np.var(train_rainfall_mm),np.var(train_crop_yield_hectare))
print("Test std: rainfall (in mm) v/s cropy yield (in hectare)",np.std(test_rainfall_mm),np.std(test_crop_yield_hectare))
print("Train std: rainfall (in mm) v/s cropy yield (in hectare)",np.std(train_rainfall_mm),np.std(train_crop_yield_hectare))


