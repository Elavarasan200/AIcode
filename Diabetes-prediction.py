import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from numpy.matrixlib.defmatrix import matrix
warnings.filterwarnings('ignore')

# Define a list of relevant features
relevant_features = ['Glucose', 'BloodPressure', 'BMI', 'Age']

# Subset the dataset to include relevant features
diabetes_dataset = pd.read_csv('diabetes.csv')
dataset = diabetes_dataset

diabetes_dataset = diabetes_dataset[relevant_features + ['Outcome']]

diabetes_dataset.duplicated()

X = diabetes_dataset[relevant_features]
Y = diabetes_dataset['Outcome']


# printing the first 5 rows of the dataset
diabetes_dataset.head()

# number of rows and Columns in this dataset
diabetes_dataset.shape
print(diabetes_dataset.info())  # Get information about data types and missing values

# getting the statistical measures of the data
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(diabetes_dataset)

na_cols = missing_values_table(diabetes_dataset, True)

diabetes_dataset.groupby('Outcome').mean()

correlation_matrix = diabetes_dataset.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Categorical columns
cat_col = [col for col in diabetes_dataset.columns if diabetes_dataset[col].dtype == 'object']
print('Categorical columns :',cat_col)
# Numerical columns
num_col = [col for col in diabetes_dataset.columns if diabetes_dataset[col].dtype != 'object']
print('Numerical columns :',num_col)

diabetes_dataset[cat_col].nunique()

round((diabetes_dataset.isnull().sum()/diabetes_dataset.shape[0])*100,2)

# calculate summary statistics
mean = diabetes_dataset['Age'].mean()
std = diabetes_dataset['Age'].std()

# Calculate the lower and upper bounds
lower_bound = mean - std*2
upper_bound = mean + std*2

print('Lower Bound :',lower_bound)
print('Upper Bound :',upper_bound)

# Drop the outliers
df4 = diabetes_dataset[(diabetes_dataset['Age'] >= lower_bound)
				& (diabetes_dataset['Age'] <= upper_bound)]

print(X)
print(Y)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)
X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
# Make predictions on the training and testing sets
Y_train_pred = classifier.predict(X_train)
Y_test_pred = classifier.predict(X_test)

# Calculate accuracy scores
training_data_accuracy = accuracy_score(Y_train_pred, Y_train)
test_data_accuracy = accuracy_score(Y_test_pred, Y_test)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
# Create a bar chart to visualize accuracies
accuracies = [training_data_accuracy, test_data_accuracy]
labels = ['Training Data', 'Test Data']


model=SVC()
model.fit(X_train , Y_train)
y_pred=model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

# Create a Logistic Regression model
logistic_regression = LogisticRegression()

# Define a parameter grid to search over
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization penalty
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],  # Solver algorithm
}
# Create a GridSearchCV object with cross-validation
grid_search = GridSearchCV(logistic_regression, param_grid, cv=5, scoring='accuracy')

# Perform the grid search to find the best hyperparameters
grid_search.fit(X, Y)

# Print the best hyperparameters and the corresponding accuracy
best_params = grid_search.best_params_
best_accuracy = grid_search.best_score_
print(f"Best Hyperparameters: {best_params}")
print(f"Best Accuracy: {best_accuracy}")

plt.figure(figsize=(8, 6))
plt.bar(labels, accuracies, width=0.4, align='center', alpha=0.5, color=['blue', 'green'])
plt.xlabel('Data Split')
plt.ylabel('Accuracy')
plt.title('Training and Test Accuracies')
plt.show()

# Data visualization section: Histogram of the "Glucose" feature
plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=20, color='blue', alpha=0.7)
plt.xlabel('Glucose Level')
plt.ylabel('Frequency')
plt.title('Distribution of Glucose Levels')

plt.show()

plt.boxplot(diabetes_dataset['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

diabetes_dataset['Age'].plot.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

plt.scatter(diabetes_dataset['Glucose'], diabetes_dataset['BMI'])
plt.title('Scatter Plot')
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()



# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Define the hyperparameter grid to search
param_grid = {
    'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],    # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required to be at a leaf node
}

# Create a GridSearchCV object with cross-validation
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
# Fit the grid search to the data
grid_search.fit(X, Y)

# Print the best hyperparameters and their corresponding accuracy score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", best_score)

# Visualizing Kernel Density Estimator for each feature
features = diabetes_dataset.columns[:-1]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.subplots_adjust(wspace=0.4, hspace=0.4)

for i, feature in enumerate(features):
    sns.kdeplot(diabetes_dataset[feature], ax=axes[i//4, i%4],shade='fill')

plt.show()


# Make predictions on a sample input data point
input_data = np.array([148,72,33.6,50]).reshape(1, -1)
input_data = scaler.transform(input_data)
prediction = classifier.predict(input_data)

print('Accuracy score of the test data : ', test_data_accuracy)
input_data = (148,72,33.6,50)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

# Number of synthetic samples to generate
num_samples_to_generate = 500

# Initialize lists to store synthetic data
synthetic_data = []
synthetic_labels = []

for _ in range(num_samples_to_generate):
    # Randomly select an index from the real data
    random_index = np.random.randint(0, len(X))

    # Select a real data point and its label
    real_data_point = X[random_index]
    real_label = Y[random_index]

    # Create a slightly modified version of the real data point
    modified_data_point = real_data_point + np.random.normal(0, 0.1, size=real_data_point.shape)

    # Append the modified data point and its label to the synthetic data
    synthetic_data.append(modified_data_point)
    synthetic_labels.append(real_label)

# Combine real and synthetic data
X_synthetic = np.vstack([X, np.array(synthetic_data)])
Y_synthetic = np.concatenate([Y, np.array(synthetic_labels)])

# Print the first 5 samples of the synthetic data
print("Synthetic Data (X_synthetic):")
print(X_synthetic[:5])

# Print the corresponding labels for the first 5 samples
print("Synthetic Labels (Y_synthetic):")
print(Y_synthetic[:5])

# Make predictions on the test set
Y_test_pred = classifier.predict(X_test)
# Calculate various performance metrics
accuracy = accuracy_score(Y_test, Y_test_pred)
precision = precision_score(Y_test, Y_test_pred)
recall = recall_score(Y_test, Y_test_pred)
f1 = f1_score(Y_test, Y_test_pred)

# Calculate various performance metrics
accuracy = accuracy_score(Y_test, Y_test_pred)
precision = precision_score(Y_test, Y_test_pred)
recall = recall_score(Y_test, Y_test_pred)
f1 = f1_score(Y_test, Y_test_pred)

# Calculate ROC-AUC score and plot ROC curve
y_scores = classifier.decision_function(X_test)
roc_auc = roc_auc_score(Y_test, y_scores)
fpr, tpr, thresholds = roc_curve(Y_test, y_scores)

# Print the performance metrics confusion matrix
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("ROC-AUC Score: {:.2f}".format(roc_auc))
print("Classification Report is:",classification_report(Y_test, Y_test_pred))
cm=confusion_matrix(Y_test, Y_test_pred)
print("Confusion matrix is:",cm)
color = 'white'
matrix = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=classifier.classes_)
matrix.plot()
plt.show()


# Create a confusion matrix
conf_matrix = confusion_matrix(Y_test, Y_test_pred)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
sc=StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

gbc=GradientBoostingClassifier(n_estimators=500,learning_rate=0.05,random_state=100,max_features=5 )

gbc.fit(x_train,y_train)

print(confusion_matrix(y_test, gbc.predict(x_test)))

print("GBC accuracy is %2.2f" % accuracy_score(
     y_test, gbc.predict(x_test)))

pred=gbc.predict(x_test)

print(classification_report(y_test, pred))

grid = {
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':np.arange(100,500,100),
}

gb = GradientBoostingClassifier()

gb_cv = GridSearchCV(gb, grid, cv = 4)

gb_cv.fit(x_train,y_train)

print("Best Parameters:",gb_cv.best_params_)

print("Train Score:",gb_cv.best_score_)

print("Test Score:",gb_cv.score(x_test,y_test))

grid = {
    'max_depth':[2,3,4,5,6,7],
}

gb = GradientBoostingClassifier()

gb_cv = GridSearchCV(gb, grid, cv = 4)

gb_cv.fit(x_train,y_train)

print("Best Parameters:",gb_cv.best_params_)

print("Train Score:",gb_cv.best_score_)

print("Test Score:",gb_cv.score(x_test,y_test))
