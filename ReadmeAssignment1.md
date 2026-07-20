The dataset requires you to make predictions on the price of a flight ticket given a set of features. The features are listed below.

# Files
train.csv - The training set which contains the features and the target
test.csv - The test set for which the target column is hidden
sample_submission.csv - A sample submission file in the correct format


# Column Description
airline: Name of the Airline company
flight: Information about flight code
source: City from which the flight takes off
departure: Time period at which the flight takes off
stops: Number of stops between source and destination cities
arrival: time period at which the flight lands
destination: City where the flight lands
class: Information about seat class
duration: Overall time taken to travel between cities in hours
days_left: Number of days between booking and trip dates
price: Information about ticket price

# Evaluation
Submissions are evaluated on r2_score between the observed target and the predicted target.

# Submission File
For each ID in the test set, you must predict a probability for the TARGET variable. The file should contain a header and have the following format:

id,price
0,2000
1,13500
2,6540
etc.

# Assignment 1
The dataset requires you to make predictions on the price of a flight ticket given a set of features. You will be provided with a training dataset and a test dataset. The labels for the test dataset will remain hidden and your task is to submit predictions for the same. The assignment will be conducted on the Kaggle Platform

# Peer Review Rubrics
Criteria


1. Identify data types of different columns
The data types of the different columns are identified and explicitly stated in the notebook


2. Present descriptive statistics of numerical columns
Details such as min value, max value, mean and median for each numerical column is presented

3.Identify and handle the missing values
Missing values are identified and are dropped or imputed

4.Identify and handle duplicates
Duplicates are identified and are dropped if they exist

5.Identify and handle outliers
Outliers are identified and explanation for retaining / dropping is provided

6.Present at least three visualizations and provide insights for the same
At least three visualizations on the data is presented 

7.Scale Numerical features and Encode Categorical features
Explanation for scaling (or not scaling) and encoding (or not encoding) is provided

8.Model Building (at least 7)
7 different types of models are trained on the data

9. Hyperparameter Tuning on any 3 of the models

Hyperparameter tuning is done on 3 of the models


10. Comparison of model performances

Performance of the models on validation set is compared