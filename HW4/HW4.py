"""
Question 1

Take the given Tuple (Systems, 26_30, 46k-50k)

put into formula

P(junior) = 113/165 ~~ 0.685
P(senior) = 52/165 ~~ 0.315

P(Systems|Junior) = 23/113 ~~ 0.204
P(26_30|Junior) = 49/113 ~~ 0.434
P(Salary|Junior) = 43/113 ~~ 0.381

P(Systems|Senior) = 8/52 ~~ 0.154
P(26_30|Senior) = 0/52 = 0
P(Salary|Senior) = 40/52 ~~ 0.769

P(Junior) = 0.685 * 0.204 * 0.434 * 0.381 ~~ 0.023

P(Senior) = 0.315 * 0.154 * 0 * 0.769 = 0

Due to the age probability making the probability of this being a senior 0 the Naive Bayes Formula would classify this tuple as a junior

"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

#Load the data
df = pd.read_csv("diabetes_train.csv")

#Split the data
train_df = df.iloc[:-10]  # all except last 10
test_df = df.iloc[-10:]   # last 10 rows

#Separate features and labels
X_train = train_df.drop("class", axis=1)
y_train = train_df["class"]

X_test = test_df.drop("class", axis=1)
y_test = test_df["class"]  # optional: to compare actual vs predicted

#Scale the features (SVMs work best with scaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train the SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')  # you can experiment with parameters
svm.fit(X_train_scaled, y_train)

#Predict on test set
predictions = svm.predict(X_test_scaled)

#Output predictions
print("Predicted class for test set:")
for i, (pred, actual) in enumerate(zip(predictions, y_test), start=1):
    print(f"Test Row {i}: Prediction = {pred}, Actual = {actual}")

"""
Question 2
Predicted class for test set:
Test Row 1: Prediction = tested_positive, Actual = tested_positive
Test Row 2: Prediction = tested_negative, Actual = tested_positive
Test Row 3: Prediction = tested_positive, Actual = tested_positive
Test Row 4: Prediction = tested_negative, Actual = tested_negative
Test Row 5: Prediction = tested_negative, Actual = tested_negative
Test Row 6: Prediction = tested_negative, Actual = tested_positive
Test Row 7: Prediction = tested_positive, Actual = tested_positive
Test Row 8: Prediction = tested_positive, Actual = tested_positive
Test Row 9: Prediction = tested_negative, Actual = tested_negative
Test Row 10: Prediction = tested_negative, Actual = tested_positive
"""

"""
Question 3 

A.
Initial Centers
H = (4, 9)
C = (8, 4)

Distance to H
A = |2 - 4| + |10 - 9| = 2 + 1 = 3 
B = |2 - 4| + |5 - 9| = 2 + 4 = 6
C = |8 - 4| + |4 - 9| = 4 + 5 = 9
D = |5 - 4| + |8 - 9| = 1 + 1 = 2
E = |7 - 4| + |5 - 9| = 3 + 4 = 7
F = |6 - 4| + |4 - 9| = 2 + 5 = 7
G = |1 - 4| + |2 - 9| = 3 + 7 = 10
h = 0

Distance to c
A = |2 - 8| + |10 - 4| = 6 + 6 = 12 
B = |2 - 8| + |5 - 4| = 6 + 1 = 7
C = 0
D = |5 - 8| + |8 - 4| = 3 + 4 = 7
E = |7 - 8| + |5 - 4| = 1 + 1 = 2
F = |6 - 8| + |4 - 4| = 2 + 0 = 2
G = |1 - 8| + |2 - 4| = 7 + 2 = 9
h = |4 - 8| + |9 - 4| = 4 + 5 = 9

Assigned Cluster

A = H
B = H
C = C
D = H
E = C
F = C
G = C
H = H

Cluster 1 with H center = A, B, D, H
Cluster 2 with C center = C, E, F, G

Compute New Centers
Mean of Cluster 1 X = (2 + 2 + 5 + 4) / 4 = 3.25
Mean of Cluster 1 Y = (10 + 5 + 8 + 9) / 4 = 8
Mean of Cluster 2 X = (8 + 7 + 6 + 1) / 4 = 5.5
Mean of Cluster 2 Y = (4 + 5 + 4 + 2) / 4 = 3.75

New centers
Cluster 1 = (3.25, 8)
Cluster 2 = (5.5, 3.75)

B.

Distance to (3.25, 8)
A = |2 - 3.25| + |10 - 8| = 1.25 + 2 = 3.25  
B = |2 - 3.25| + |5 - 8| = 1.25 + 3 = 4.25  
C = |8 - 3.25| + |4 - 8| = 4.75 + 4 = 8.75  
D = |5 - 3.25| + |8 - 8| = 1.75 + 0 = 1.75  
E = |7 - 3.25| + |5 - 8| = 3.75 + 3 = 6.75  
F = |6 - 3.25| + |4 - 8| = 2.75 + 4 = 6.75  
G = |1 - 3.25| + |2 - 8| = 2.25 + 6 = 8.25  
H = |4 - 3.25| + |9 - 8| = 0.75 + 1 = 1.75

Distance to (5.5, 3.75)
A = |2 - 5.5| + |10 - 3.75| = 3.5 + 6.25 = 9.75  
B = |2 - 5.5| + |5 - 3.75| = 3.5 + 1.25 = 4.75  
C = |8 - 5.5| + |4 - 3.75| = 2.5 + 0.25 = 2.75  
D = |5 - 5.5| + |8 - 3.75| = 0.5 + 4.25 = 4.75  
E = |7 - 5.5| + |5 - 3.75| = 1.5 + 1.25 = 2.75  
F = |6 - 5.5| + |4 - 3.75| = 0.5 + 0.25 = 0.75  
G = |1 - 5.5| + |2 - 3.75| = 4.5 + 1.75 = 6.25  
H = |4 - 5.5| + |9 - 3.75| = 1.5 + 5.25 = 6.75

Assigned Cluster After Reassignment:
A = Cluster 1
B = Cluster 1
C = Cluster 2
D = Cluster 1
E = Cluster 2
F = Cluster 2
G = Cluster 2
H = Cluster 1

Cluster 1 with center (3.25, 8) = A, B, D, H
Cluster 2 with center (5.5, 3.75) = C, E, F, G

Question 4

A. In PNG

B.

A = B, D, H
B = A, C, D, E, F, G, H
C = B, D, E, F, H
D = A, B, C, E, F, H
E = B, C, D, F, H
F = B, C, D, E, G, H
G = B, F
H = A, B, C, D, E, F

Count
A = 4
B = 8
C = 6
D = 7
E = 6
F = 7
G = 3
H = 7

this means every point is a core point
"""