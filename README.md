# Introduction

Hello, and welcome to our research on hotel reservation cancellations. This project is done for our ECS 171 - Machine Learning class at UC Davis, instructed by Edwin Solares. Throughout our project, we demonstrate how we are able to accurately predict future hotel cancellations, utilizing recorded data of previous behaviors and various machine learning tools. This could possibly spark popularity amongst denying reservations to hotels, which is something popular companies like AirBNB utilizes.  Our results from the data holds significance as we are able positively impact hotels from all over the world by helping them save their time, energy, and resources on tourists and guests who are more likely to show up.

## Methods

This section will provide the different methods and means for how we were going to possibly predict hotel cancellations.

Links to the Following Sections and Where They Apply:
1) [Data Exploration](Assignments/Data%20Exploration%20Milestone/DataExploration.ipynb)
2) [Preprocessing, Data Splitting, Model One, Reporting Results](Assignments/Preprocessing%20&%20First%20Model/PreprocessingFirstModelMilestone.ipynb)
3) [Model Two, Reporting Results](Assignments/Second%20Model/Second%20Model.ipynb)

### Data Exploration

Our data exploration displayed our hotel reservations data in 3 separate cells. Our first cell contained information on our whole dataframe. Our second cell contained information on our attributes. Our third cell contained information on our class. These blocks exhibited details on our dataframes, including their shape, their total number of observations, a list of column names, and the data type in each column. 

```
#var = df or X or y
print("Shape:        ", var.shape)
print("Total Observations: ", var.shape[0] * var.shape[1])
cols = list(var.columns)
cols_dtype = {}
for i in range(0, len(cols)):
    cols_dtype[cols[i]] = df[cols[i]].dtype
print("var Column Desc:  ", cols_dtype)
```
Afterwards, we found columns containing missing data.
```
cols = list(var.columns)
for i in range(len(cols)):
    print("Missing Values (" + cols[i] + "): " + str( df[cols[i]].isna().sum() ))
```
Lastly, we used seaborn pairplots, histplots, and displots to visualize our data frequencies and distributions.
```
#For numerical values
sns.pairplot(data=df, y_vars=['is_canceled'], x_vars=['x', 'x', 'x', …])
sns.histplot(data=df, x='x', 'x', 'x', …, kde=True)
#For nomail values
sns.displot(df['x'].astype(str))
print(df['x'].value_counts())
```


### Preprocessing

First, we transformed our data, removing any unnecessary or duplicate columns. For example, there was a ‘reservation_status’ attribute that was essentially equivalent to our class. 
```
df = df.drop(columns=['country', 'agent', 'company', 'reservation_status_date', 'reservation_status'])
```
Second, we imputed our data to handle NaN/Null/None values.
```
df = df.replace(np.nan, 0)
```
Third, we used keras MinMaxScaler() to normalize our numerical data between values of 0 and 1. 
```
norm_scaler = MinMaxScaler()
df[numericalAttribs] = norm_scaler.fit_transform(df[numericalAttribs])
```
Fourth, we encoded our categorical columns to numerical values between 0 ~ N - 1 unique values. 
```
encodings = {}
for a in categoricalAttribs:
    keys = {}
    iter = 0
    for val in list(df[a].unique()):
        keys[val] = iter
        iter += 1
    encodings[a] = keys

df = df.replace(encodings)
```


### Data Splitting

Before building any models, we split the Train:Test data 70:30, then took [~‘is_canceled’] as our attributes(X) and [‘is_canceled’] as our class(y). 
```
Train, Test = train_test_split(df, test_size=0.3)
X_train = Train.drop('is_canceled', axis = 1)
X_test = Test.drop('is_canceled', axis = 1)
y_train = Train['is_canceled']
y_test = Test['is_canceled']
```


### Model One

Our first model used keras Sequential() and Dense() to build an artificial neural network(ANN) model. Our ANN had an input size of 26 and activation functions of ‘relu’, ‘softmax’, ‘softmax’, ‘tanh’, ‘tanh’, ‘tanh’, ‘sigmoid’ with unit sizes of 50, 35, 35, 20, 20, 10, and 1 respectively. 
```
model = Sequential() 
model.add(Dense(units = 50, activation = 'relu', input_dim = 26))
model.add(Dense(units = 35, activation = 'softmax'))
model.add(Dense(units = 35, activation = 'softmax'))
model.add(Dense(units = 20, activation = 'tanh'))
model.add(Dense(units = 20, activation = 'tanh'))
model.add(Dense(units = 10, activation = 'tanh'))
model.add(Dense(units = 1, activation = 'sigmoid'))
```
Afterwards, we compiled and trained our ANN for 100 epochs.
```
model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy')
model.fit(X.astype('float'), y, batch_size = 1, epochs = 100)
```
Lastly, we thresholded our predicted yhat values using 0.5.
```
yhat = model.predict(X.astype(float))
yhat = [1 if y>=0.5 else 0 for y in yhat]
```


### Model Two

Our second model used sklearn SVC() to build a SVM model with both an rbf kernel or linear kernel. 
```
svm = SVC(kernel = 'rbf' or 'linear')
```
Afterwards, we trained our SVM.
```
svm.fit(X, y)
```


### Reporting Results

Lastly, for both models, we used sklearn classification_report() to compare our yhat values for accuracy against our y_train and y_test values. 
```
print(classification_report(y_test, yhat_test_model))
```
Additionally, we used sklearn PCA() and matplotlib to plot our yhat values against our y_test.
```
pca = PCA(n_components=1)
pca.fit(X_train)
X_test_final = pca.transform(X_test)
X_test_final = pd.DataFrame(X_test_final)
X_test_final = X_test_final[0]

colors = []

for i in range(len(yhat or y_test)):
    if yhat[i] == 0: 
        colors.append("blue")
    elif yhat[i] == 1: 
        colors.append("orange")

plt.scatter(X_test_final, y_test, color=colors) 
```

## Results

This section will provide the results from the methods and means listed in the section above.

Links to the Following Sections and Where They Apply:
1) [Preprocessing, Model One](Assignments/Preprocessing%20&%20First%20Model/PreprocessingFirstModelMilestone.ipynb)
2) [Model Two](Assignments/Second%20Model/Second%20Model.ipynb)

### Preprocessing
Our results from transforming was dropping the country, agent, company, reservation_status_date, and reservation_status columns.

Afterwards, we found that there is only one column with missing data, as shown below.
```
hotel                             False
is_canceled                       False
lead_time                         False
arrival_date_year                 False
arrival_date_month                False
arrival_date_week_number          False
arrival_date_day_of_month         False
stays_in_weekend_nights           False
stays_in_week_nights              False
adults                            False
children                           True
babies                            False
meal                              False
market_segment                    False
distribution_channel              False
is_repeated_guest                 False
previous_cancellations            False
previous_bookings_not_canceled    False
reserved_room_type                False
assigned_room_type                False
booking_changes                   False
deposit_type                      False
days_in_waiting_list              False
customer_type                     False
adr                               False
required_car_parking_spaces       False
total_of_special_requests         False
dtype: bool
```
Using our results, the resulting dataframe from imputing was replacing missing values in the children column with 0.

After normalizing our numerical data, the resulting columns transformed into this below:
![Numerical Data](Images/NumericalAttribs.png "Numerical Data")

After encoding our categorical columns to numerical values, the resulting columns transformed into this below:
![Categorical Data](Images/CategoricalAttribs.png "Categorical Data")

### Model One

Our results from our ANN model were measurements of 79% training accuracy and 79% testing accuracy as displayed via Classification Report.
```
2612/2612 [==============================] - 1s 430us/step
1120/1120 [==============================] - 0s 436us/step
              precision    recall  f1-score   support

           0       0.78      0.91      0.84     52506
           1       0.80      0.57      0.67     31067

    accuracy                           0.79     83573
   macro avg       0.79      0.74      0.75     83573
weighted avg       0.79      0.79      0.78     83573

              precision    recall  f1-score   support

           0       0.79      0.91      0.85     22660
           1       0.80      0.59      0.67     13157

    accuracy                           0.79     35817
   macro avg       0.79      0.75      0.76     35817
weighted avg       0.79      0.79      0.78     35817
```

Our plot using sklearn PCA and matplotlib is shown below. 
![PCA of ANN](Images/annPCA.png "PCA of ANN")

All of these results are displayed in the [Model Two Notebook](Assignments/Second%20Model/Second%20Model.ipynb).

### Model Two

Our results from our SVM model were measurements of 77% testing accuracy for a linear kernel and 75% testing accuracy for a rbf kernel. Here is the Classification Report and PCA plot for the linear kernel:
```
              precision    recall  f1-score   support

           0       0.73      1.00      0.84     22643
           1       0.98      0.38      0.54     13174

    accuracy                           0.77     35817
   macro avg       0.86      0.69      0.69     35817
weighted avg       0.82      0.77      0.73     35817
```
![PCA of Linear Kernel](Images/linearPCA.png "PCA of Linear Kernel")

Here is the Classification Report and PCA plot for the rbf kernel:
```
              precision    recall  f1-score   support

           0       0.72      1.00      0.83     22643
           1       0.99      0.33      0.49     13174

    accuracy                           0.75     35817
   macro avg       0.85      0.66      0.66     35817
weighted avg       0.82      0.75      0.71     35817
```
![PCA of RBF Kernel](Images/rbfPCA.png "PCA of RBF Kernel")

## Discussion

This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!

## Conclusion 

To conclude, we think that the project went decently well. Being able to take hotel data and predict a reservation’s cancellation status with a minimum of 75% accuracy is very satisfactory. However, we were both surprised that after our first model with the ANN, we were not able to find a model that performed better. We speculate that maybe the relationship of hotel cancellations and the different attributes provided in the dataset may not be as strong as we initially thought. We feel as if there is room for improvement for this project. There are several possible directions that we could have taken this project. If we were provided a more diverse set of attributes, preprocessed/transformed the data differently, ran the ANN for more epochs, or utilized more dense layers with different activation functions, maybe we could find a more accurate relationship that we can use to predict cancellations. However, we felt quite limited on how to improve our results with our current knowledge. We could have impacted some portions of the project, but, for the most part, we did as best as we could to build a model that accurately predicts hotel cancellations. Thus, to close, we believe that we succeeded in being able to apply many of the tools that we learned in our class to a real world dataset and predict results that have beneficial effects on companies around the world. 

## Collaboration Section
### Name: Steven Gong
#### Contribution
Spent majority of the time working on the implementation of the coding portions of each milestone like Data Exploration, Preprocessing and First Model Building, and finding a better second model. Implemented a good portion of the write up as well. 

### Name: Thaddeus Dziura
#### Contribution
Spent more time on facilitating conversations and delegating tasks on what needed to be done for each assignment. Provided feedback for coding portions and also wrote the majority of notebook explanations for people to follow along. Did some of the coding portions in the Data Exploration and Preprocessing Milestones.


# Prior Submissions
1) [Group Project Abstract Submission](Assignments/Abstraction.md)
2) [Data Exploration Milestone](Assignments/Data%20Exploration%20Milestone)
3) [Preprocessing & First Model Building and Evaluation Milestone](Assignments/Preprocessing%20%26%20First%20Model)
4) [Final Submission](README.md)

# Related Notebooks
1) [Data Exploration Notebook](Assignments/Data%20Exploration%20Milestone/DataExploration.ipynb)
2) [Preprocessing & First Model Notebook](Assignments/Preprocessing%20&%20First%20Model/PreprocessingFirstModelMilestone.ipynb)
3) [Second Model Notebook](Assignments/Second%20Model/Second%20Model.ipynb)
