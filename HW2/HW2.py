"""

Lance C Swett

Question 1

Object 1, A, Excellent, 45
Object 3, C, Good, 64

Value A != Value C = 1

Excellent = 1
Good = 0.5
Fair = 0

Excellent - Good = 0.5

Min = 22
Max = 64

Object 1
(45-22)/(64-22) = 23/42 ~~ 0.548

Object 3
(64-22)/(64-22) = 42/42 = 1

1 - 0.548 ~~ 0.452

Total Distance

(1 + 0.5 + 0.452)/3 ~~0.651

"""

# Question 2

import math

def manhattan_distance(vec1, vec2):
    return sum(abs(a - b) for a, b in zip(vec1, vec2))

def euclidean_distance(vec1, vec2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

# Example vectors
v1 = [1, 2, 3]
v2 = [4, 0, 8]

# Output results
print("Manhattan Distance:", manhattan_distance(v1, v2))
print("Euclidean Distance:", euclidean_distance(v1, v2))

"""
Question 3

Calculate expected values

Attended-Pass
31 * 33 / 54 ~~ 18.94

Attended-Fail
31 * 21 / 54 ~~ 12.06

Skipped-Pass
23 * 33 / 54 ~~ 14.06

Skipped-Fail
23 * 21 / 54 ~~ 8.94

Compute Statistic

(Observed - Expected) ^ 2 / Expected

Attended-Pass
(6.06) ^ 2 / 18.94 ~~ 1.94

Attended-Fail
(-6.06) ^ 2 / 12.06 ~~ 3.05

Skipped-Pass
(-6.06) ^ 2 / 14.06 ~~ 2.61

Skipped-Fail
(6.06) ^ 2 / 8.94 ~~ 4.11


Add together

x ^ 2 = 11.71

Degrees of Freedom

df = (rows - 1) * (columns - 1) = 1

a = 0.05

Critical value = 3.841

11.71 > 3.841 We reject the idea that passing a class is dependant on attending it

"""

"""
Question 4

# Calculate and print correlation
correlation <- cor(mtcars$mpg, mtcars$wt)
print(paste("Correlation between mpg and wt:", correlation))

# Create the plot
plot(mtcars$wt, mtcars$mpg,
     main = "MPG vs Weight",
     xlab = "Car Weight (1000 lbs)",
     ylab = "Miles Per Gallon",
     pch = 19,
     col = "blue")

"""

#Question 5

import pandas as pd

# Load the dataset
df = pd.read_csv("metabolite.csv")

# drop columns with > 75% missing values
threshold = 0.75
df_cleaned = df.loc[:, df.isnull().mean() < threshold]

# impute missing values with the column median
df_imputed = df_cleaned.fillna(df_cleaned.median(numeric_only=True))

# View the cleaned dataset
print(f"Original shape: {df.shape}")
print(f"After cleaning: {df_imputed.shape}")


#Question 6

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Separate the label column
labels = df_imputed['Label']
df_features = df_imputed.drop(columns=['Label'])

# Standardize the features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Apply PCA to reduce to 2 principal components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled)

# Create a DataFrame with PCA results and labels
pca_df = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
pca_df['Label'] = labels

# Plot the PCA results with points colored by label
plt.figure(figsize=(8,6))
for label in pca_df['Label'].unique():
    subset = pca_df[pca_df['Label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=label, alpha=0.7)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Metabolite Data (Colored by Label)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_plot.png")
