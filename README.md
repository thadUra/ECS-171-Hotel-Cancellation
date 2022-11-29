# ECS-171-Hotel-Cancellation
For our final project in ECS 171 - Machine Learning, we are going to explore and use machine learning tools to predict if a hotel reservation will be cancelled.

## Finish major preprocessing
1. Impute Data: To handle NaN/Null/None values in our data, We used dropna() to drop missing data.

2. Encode Data: To handle categorical data, we encoded our categorical columns of values between 0 ~ N - 1 unique values. 
    
    - Our categorical attributes included: 
                    ['hotel',
                    'arrival_date_month',
                    'arrival_date_year',
                    'arrival_date_week_number',
                    'arrival_date_day_of_month',
                    'meal',
                    'market_segment',
                    'distribution_channel',
                    'is_repeated_guest',
                    'reserved_room_type',
                    'assigned_room_type',
                    'deposit_type',
                    'customer_type',
                    'required_car_parking_spaces',
                    'reservation_status']

3. Scale Data: Normalize Data: between 0 and 1 using keras MinMaxScaler(), Standardize Data: between -1 and 1 using keras StandardScaler()

    - Our scaled attributes included: 
                    ['lead_time', 
                    'stays_in_week_nights', 
                    'stays_in_weekend_nights', 
                    'adults',
                    'children',
                    'babies',
                    'previous_cancellations',
                    'previous_bookings_not_canceled',
                    'booking_changes',
                    'days_in_waiting_list',
                    'adr',
                    'total_of_special_requests']

    - Normalizing and Standardizing will help us achieve a faster Gradient Descent Algo

4. No Data Expansion, as we had a large enough dataset

## Train your first model
1. Split Train and Test data 70:30 for cross validation
2. X's and y's were is_canceled col. and ~is_canceled col.
3. Classification Models to chose from: NN, NB, SVM, Log Reg, KNN, DTL
4. We chose our first model as a somewhat simple Sequential Neural Network
5. First we fit the model, Then, we predicted yhat for X_train and X_test

## Evaluate your model compare training vs test error
1. Using classification_report from sklearn, we saw that our model underfitting whether a hotel will be canceled based on previous user tendencies. 
2. The precision, recall, f1-score, and accuracy were all very similar when comparing the y's with the yhat's.

## Where does your model fit in the fitting graph
1. We analyzed the classification_report for evaluating our error to conclude that our data was underfitting(not overfit or correctly fit)