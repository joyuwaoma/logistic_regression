# logistic_regression
![LgR](https://github.com/joyuwaoma/logistic_regression/blob/main/LgR.png)

## Objective:
1. Use Logistic Regression for classification
2. Preprocess data for modeling
3. Implement Logistic regression on real-world data

## Scenario:
Assume that you are working for a telecommunications company that is concerned about the number of customers leaving their land-line business for cable competitors. They need to understand who is more likely to leave the company.

## Load the Telco Churn data:
Telco Churn is a hypothetical data file that concerns a telecommunications company's efforts to reduce turnover in its customer base. Each case corresponds to a separate customer, and it records various demographic and service usage information. Before you can work with the data, you must use the URL to get the ChurnData.csv.

## About the dataset:
I will use a telecommunications dataset for predicting customer churn. This is a historical customer dataset where each row represents one customer. The data is relatively easy to understand, and I may uncover insights I can use immediately. Typically, it is less expensive to keep customers than acquire new ones, so this analysis focuses on predicting the customers who will stay with the company. 

This data set provides you with information about customer preferences (age, gender, marital status, etc.), services opted for, personal details, etc., which helps you predict customer churn.

## Data Preprocessing:
After loading the dataset, select some features for the modeling. Also, we change the target data type to be an integer, as it is a requirement by the scikit-learn algorithm.
For this project, we can use a subset of the fields available to develop our model. Let us assume that the fields we use are 'tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', and of course 'churn'.

```python
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df
```

For modeling the input fields X and the target field y need to be fixed. Since the target to be predicted is 'churn', the data under this field will be stored under the variable 'y'. We may use any combination or all of the remaining fields as the input. Store these values in the variable 'X'.
```python
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]  #print the first 5 values
```

```python
y = np.asarray(churn_df['churn'])
y[0:5] #print the first 5 values
```

It is also a norm to standardize or normalize the dataset in order to have all the features at the same scale. This helps the model learn faster and improves the model performance. We may make use of StandardScalar function in the Scikit-Learn library.
```python
X_norm = StandardScaler().fit(X).transform(X)
X_norm[0:5]
```

## Splitting the dataset:
The trained model has to be tested and evaluated on data which has not been used during training. Therefore, it is required to separate a part of the data for testing and the remaining for training. For this, we may make use of the train_test_split function in the scikit-learn library.


