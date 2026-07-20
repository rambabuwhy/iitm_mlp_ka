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