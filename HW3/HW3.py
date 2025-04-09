#Question 2

import numpy as np

ages = np.array([13, 15, 16, 16, 19, 20, 20, 21, 22, 22, 25, 25, 25, 25, 30, 33, 33, 35, 35, 35, 35, 36, 40, 45, 46, 52, 70])

def min_max_normalize(data, new_min, new_max):
    old_min = np.min(data)
    old_max = np.max(data)
    normalized = (data - old_min) / (old_max - old_min)
    return normalized * (new_max - new_min) + new_min

normalized_ages = min_max_normalize(ages, 0, 1)

print("Normalized Ages:", normalized_ages)



#Question 3

from collections import defaultdict

#recreate the table
data = [
    {'department': 'sales',     'age': '31_35', 'salary': '46K_50K', 'status': 'senior', 'count': 30},
    {'department': 'sales',     'age': '26_30', 'salary': '26K_30K', 'status': 'junior', 'count': 40},
    {'department': 'sales',     'age': '31_35', 'salary': '31K_35K', 'status': 'junior', 'count': 40},
    {'department': 'systems',   'age': '21_25', 'salary': '46K_50K', 'status': 'junior', 'count': 20},
    {'department': 'systems',   'age': '31_35', 'salary': '66K_70K', 'status': 'senior', 'count': 5},
    {'department': 'systems',   'age': '26_30', 'salary': '46K_50K', 'status': 'junior', 'count': 3},
    {'department': 'systems',   'age': '41_45', 'salary': '66K_70K', 'status': 'senior', 'count': 3},
    {'department': 'marketing', 'age': '36_40', 'salary': '46K_50K', 'status': 'senior', 'count': 10},
    {'department': 'marketing', 'age': '31_35', 'salary': '41K_45K', 'status': 'junior', 'count': 4},
    {'department': 'secretary', 'age': '46_50', 'salary': '36K_40K', 'status': 'senior', 'count': 4},
    {'department': 'secretary', 'age': '26_30', 'salary': '26K_30K', 'status': 'junior', 'count': 6},
]

#data = [row for row in data if row['salary'] == '46K_50K']

#Grab information from table for attribute
def group_by_attribute(data, attribute):
    grouped = defaultdict(lambda: {'junior': 0, 'senior': 0})
    for row in data:
        key = row[attribute]
        grouped[key][row['status']] += row['count']
    return grouped

def entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)

def info_gain(total_counts, grouped_counts):
    total_entropy = entropy(total_counts)
    total = sum(total_counts)
    weighted_entropy = 0
    for subgroup in grouped_counts.values():
        group_total = sum(subgroup.values())
        group_entropy = entropy([subgroup['junior'], subgroup['senior']])
        weighted_entropy += (group_total / total) * group_entropy
    return total_entropy - weighted_entropy

total_junior = sum(row['count'] for row in data if row['status'] == 'junior')
total_senior = sum(row['count'] for row in data if row['status'] == 'senior')
total_counts = [total_junior, total_senior]

print(f"Total Junior: {total_junior}, Total Senior: {total_senior}")
print(f"Total Entropy: {entropy(total_counts):.4f}\n")

for attr in ['department', 'age', 'salary']:
    grouped = group_by_attribute(data, attr)
    ig = info_gain(total_counts, grouped)
    print(f"Information Gain for {attr}: {ig:.4f}")



"""
For the first level it looks like I would split by salary first as that gives the most info gain

For the Second level i think splitting by either is fine as they are both perfect splits by the data we have, even if in the real world this data is likely incorrect
"""
#Question 4

def classify_status(salary, department=None):
    if salary == "26K_30K":
        return "junior"
    elif salary == "31K_35K":
        return "junior"
    elif salary == "36K_40K":
        return "senior"
    elif salary == "41K_45K":
        return "junior"
    elif salary == "66K_70K":
        return "senior"
    elif salary == "46K_50K":
        if department == "sales":
            return "senior"
        elif department == "systems":
            return "junior"
        elif department == "marketing":
            return "senior"
        else:
            return "unknown"  # in case department is missing or unrecognized
    else:
        return "unknown"  # salary not in known categories
