# Assignment 1 - TOPSIS

# Name: Hrsh Dhingra
# Roll no. 102103443

import pandas as pd 
import numpy as np
import sys


# Step 1: Checking that command line input is correct

def topsis():
    if len(sys.argv) != 5:
        print("Incorrect number of command line parameters entered! Try again")
        sys.exit(1)

    try:
        inputFile = sys.argv[1]
        df = pd.read_csv(inputFile)
    except FileNotFoundError:
        print("File not Found! Try again")
        sys.exit(1)
        
    try:    
        weights = list(map(float, sys.argv[2].strip().split(',')))
        impacts = list(sys.argv[3].strip().split(','))
    except:
        print("Separate weights and impacts with commas! Try again")
        sys.exit(1)
        
    resultFile = sys.argv[4]

    if(df.shape[1]< 3):
        print("Input file must contain three or more columns! Try again")
        sys.exit(1)
            
    if(( len(weights) == len(impacts) == (df.shape[1]-1)) == False):
        print("Number of weights, number of impacts and number of columns must be same! Try again")
        sys.exit(1)

    for imp in impacts:
        if(imp=='+' or imp=='-'):
            continue
        else:
            print("Impacts must be either positive or negative! Try again")
            sys.exit(1)

    cat_features=[i for i in df.columns[1:] if df.dtypes[i]=='object']
    if(len(cat_features)!=0):
        print("Second to last columns must contain numeric values only! Try again")
        sys.exit(1)

feature_values = df.iloc[:,1:].values
feature_values

options = df.iloc[:,0].values
options

# Step 2: Convert Categorical to Numerical

# Step 3 : Vector Normalization

sum_cols=[0]*len(feature_values[0])
sum_cols   

#Calculating root of sum of squares
for i in range(len(feature_values)):
    for j in range(len(feature_values[i])):
        sum_cols[j]+=np.square(feature_values[i][j])
            
   for i in range(len(sum_cols)):
    sum_cols[i]=np.sqrt(sum_cols[i])

#Normalized Decision Matrix
for i in range(len(feature_values)):
    for j in range(len(feature_values[i])):
        feature_values[i][j]=feature_values[i][j]/sum_cols[j]


#Step 4: Weight Assignment

weighted_feature_values=[]
for i in range(len(feature_values)):
    temp=[]
    for j in range(len(feature_values[i])):
        temp.append(feature_values[i][j]*weights[j])
    weighted_feature_values.append(temp)

weighted_feature_values = np.array(weighted_feature_values)

# Step 5: Find Ideal Best and Ideal Worst

VjPos=[]
VjNeg=[]
for i in range(len(weighted_feature_values[0])):
    VjPos.append(weighted_feature_values[0][i])
    VjNeg.append(weighted_feature_values[0][i])

# Calculating values of Ideal worst and Ideal best arrays
for i in range(1,len(weighted_feature_values)):
    for j in range(len(weighted_feature_values[i])):
        if impacts[j]=='+':
            if weighted_feature_values[i][j]>VjPos[j]:
                VjPos[j]=weighted_feature_values[i][j]
            elif weighted_feature_values[i][j]<VjNeg[j]:
                VjNeg[j]=weighted_feature_values[i][j]
        elif impacts[j]=='-':
            if weighted_feature_values[i][j]<VjPos[j]:
                VjPos[j]=weighted_feature_values[i][j]
            elif weighted_feature_values[i][j]>VjNeg[j]:
                VjNeg[j]=weighted_feature_values[i][j]


# Step 6: Calculate Euclidean distance

Sjpositive=[0]*len(weighted_feature_values)
Sjnegative=[0]*len(weighted_feature_values)
for i in range(len(weighted_feature_values)):
    for j in range(len(weighted_feature_values[i])):
        Sjpositive[i]+=np.square(weighted_feature_values[i][j]-VjPos[j])
        Sjnegative[i]+=np.square(weighted_feature_values[i][j]-VjNeg[j])

for i in range(len(Sjpositive)):
    Sjpositive[i]=np.sqrt(Sjpositive[i])
    Sjnegative[i]=np.sqrt(Sjnegative[i])

Sjpositive
Sjnegative

performance_score=[0]*len(weighted_feature_values)
for i in range(len(weighted_feature_values)):
    performance_score[i]=Sjnegative[i]/(Sjnegative[i]+Sjpositive[i])

performance_score


# Step 8: TOPSIS Score and Rank

final_scores_sorted = np.argsort(performance_score) # this returns indices of elements in sorted order
max_index = len(final_scores_sorted)
rank = []
for i in range(len(final_scores_sorted)):
        rank.append(max_index - np.where(final_scores_sorted==i)[0][0])# since we know final_scores_sorted is already sorted, so it will need ranking from back side, so we need to subtract from maximum and get first value of tuple returned by np.where function

rank_df = pd.DataFrame({"TOPSIS Score" : performance_score, "Ranks": np.array(rank)})

rank_df

df = pd.concat([df,rank_df],axis=1)
df.to_csv(resultFile, index=False)
    
if __name__ == "__main__":
   topsis()