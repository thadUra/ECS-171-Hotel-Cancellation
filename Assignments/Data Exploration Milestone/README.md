# ECS-171-Hotel-Cancellation
For our final project in ECS 171 - Machine Learning, we are going to explore and use machine learning tools to predict if a hotel reservation will be cancelled.

## Our Preprocessing:
1. Impute Data: Handle the NaN/Null values within our data. We either want to use dropna() to drop all NaN data, or replace the NaN data with a mean value
2. Encode Data: Many of our values are strings or dates that need to be encoded into a number value like a float or integer
3. Normalize Data: Our values will are not normalized between 0 and 1, therefore, we would like the normalize the data using the MinMaxScaler from keras before taking any further action.
4. Standardize Data
5. Normalizing and Standardizing will help us achieve a faster Gradient Descent Algo. if we decide to use it in any of our models (probably yes). 