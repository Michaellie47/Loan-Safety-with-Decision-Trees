import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import scipy.stats 


%matplotlib inline
sns.set()
np.random.seed(416)


# First we load in the dataset and then inspect the values.

# Set seed for the whole program
np.random.seed(416)

# Load in data
loans = pd.read_csv('lending-club-data.csv')
loans.head()

# One of the features we will use in this assignment is the "grade" of the loan. We can investigate what this feature, "grade", looks like.
# Want the grades to show up in order from high to low
grade_order = sorted(loans['grade'].unique())

sns.countplot(x='grade', data=loans, order=grade_order)


# We can see that over half of the loan grades are assigned values A or B. Each loan is assigned one of these grades, along with a more finely discretized feature called subgrade (feel free to explore that feature column as well!). 
# These values depend on the loan application and credit report, and determine the interest rate of the loan.
# Now let's look at another feature that will be used, "home ownership".
ownership_order = sorted(loans['home_ownership'].unique())

sns.countplot(x='home_ownership', data=loans, order=ownership_order)

# The target column (label column) of the dataset that we are interested in is called bad_loans. In this column 1 means a risky (bad) loan 0 means a safe loan.
# To make it more intuitive, we reassign target to be +1 as safe loan and -1 as bad loan.
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)

# Drop the old bad_loans column
loans = loans.drop(columns='bad_loans')

# Now, let's explore the distribution of values for safe_loans. This gives us a sense of how many safe and risky loans are present in the dataset.
only_safe = loans[loans['safe_loans'] == 1]
only_bad = loans[loans['safe_loans'] == -1]

print(f'Number safe  loans: {len(only_safe)} ({len(only_safe) * 100.0 / len(loans):.2f}%)')
print(f'Number risky loans: {len(only_bad)} ({len(only_bad) * 100.0 / len(loans):.2f}%)')

# Now, find the most frequent grade.
most_frequent_grade = loans['grade'].value_counts().idxmax()
mode_grade = most_frequent_grade
print(mode_grade)

# Now, find the percentage of loans for rent.
print(loans.columns)
percent_rent = (loans['home_ownership'] == 'RENT').mean()
print(percent_rent)

# We will be using both numeric and categorical features to predict if a loan is safe or risky. 
# The features are described in code commments in the next cell.

features = [
    'grade',                     # grade of the loan (e.g. A or B)
    'sub_grade',                 # sub-grade of the loan (e.g. A1, A2, B1)
    'short_emp',                 # one year or less of employment (0 or 1)
    'emp_length_num',            # number of years of employment (a number)
    'home_ownership',            # home_ownership status (one of own, mortgage, rent or other)
    'dti',                       # debt to income ratio (a number)
    'purpose',                   # the purpose of the loan (one of many values)
    'term',                      # the term of the loan (36 months or 60 months)
    'last_delinq_none',          # has borrower had a delinquincy (0 or 1)
    'last_major_derog_none',     # has borrower had 90 day or worse rating (0 or 1)
    'revol_util',                # percent of available credit being used (number between 0 and 100)
    'total_rec_late_fee',        # total late fees received to day (a number)
]

target = 'safe_loans'                   # prediction target (y) (+1 means safe, -1 is risky)

# Extract the feature columns and target column
loans = loans[features + [target]]
loans.head()

loans.columns

# Utilizing get_dummies() method for one-hot encoding dummy variable.
loans = pd.get_dummies(loans)
features = list(loans.columns)
features.remove('safe_loans')
features

# Below is the preview of the new dataset.
loans.head()


# Now we will start by splitting the data into 80% training dataset and 20% validation dataset.
train_data, validation_data = train_test_split(loans, test_size=0.2)

# Now, using the sklearn's DecisionTreeClassifier to train the model on the training dataset.
# Train Decision Tree with max depth = 6
dt = DecisionTreeClassifier(max_depth = 6)
decision_tree_model = dt.fit(train_data[features], train_data[target])

# Let's visualize the tree below,
from sklearn import tree


def draw_tree(tree_model, features):
    """
    visualizes a Decision Tree
    """
    tree_data = tree.export_graphviz(tree_model, 
                                    impurity=False, 
                                    feature_names=features,
                                    class_names=tree_model.classes_.astype(str),
                                    filled=True,
                                    out_file=None)
    graph = graphviz.Source(tree_data) 
    display(graph)
    
small_tree_model = DecisionTreeClassifier(max_depth=2, random_state=0)
small_tree_model.fit(train_data[features], train_data[target])
draw_tree(small_tree_model, features)


# Below we will compute the training accuracy and validation accuracy using the decision tree model above
# Predicting on training set
pred_train = decision_tree_model.predict(train_data[features])
print(pred_train)

# Accuracy on training set
decision_train_accuracy = accuracy_score(train_data[target], pred_train)
print(decision_train_accuracy)

# Predicting on validation set
pred_val = decision_tree_model.predict(validation_data[features])
print(pred_val)

# Accuracy on validation set
decision_validation_accuracy = accuracy_score(validation_data[target], pred_val)
print(decision_validation_accuracy)


# Now we want to train on a Decision Tree model with a different depth, max_depth = 10, then compute the train and validation accuracies.
# Initializing big_tree_model with new depth = 10
big_tree_model = DecisionTreeClassifier(max_depth = 10)
big_tree_model.fit(train_data[features], train_data[target])

# Find the big_tree_model accuracy on training data
big_tree_pred_train = big_tree_model.predict(train_data[features])
print(big_tree_pred_train)

big_train_accuracy = accuracy_score(train_data[target], big_tree_pred_train)
print(big_train_accuracy)

# Find the big_tree_model accuracy on validation data
big_tree_pred_val = big_tree_model.predict(validation_data[features])
print(big_tree_pred_val)

big_validation_accuracy = accuracy_score(validation_data[target], big_tree_pred_val)
print(big_validation_accuracy)


# Now, lets use an optimization algroithm so the tree doesn't potentially overfit or underfit.
# Using GridSearchCV from sklearn to optimize which hyperparameter to use.

# Initializations for hyper-parameters such as min_samples_leaf and max_depth
hyperparameters= {
    'min_samples_leaf' : [1, 10, 50, 100, 200, 300],
    'max_depth' : [1, 5, 10, 15, 20]
}


search = GridSearchCV(
    estimator = DecisionTreeClassifier(random_state = 0),
    param_grid = hyperparameters,
    cv = 6,
    return_train_score = True
)

search.fit(train_data[features], train_data[target])


print(search.best_params_) # result from the GridSearchCV


# Plotting both training and validation accuracies.
def plot_scores(ax, title, search, hyperparameters, score_key):
    # Get results from GridSearch and turn scores into matrix
    cv_results = search.cv_results_
    scores = cv_results[score_key]
    scores = scores.reshape((len(hyperparameters['max_depth']), len(hyperparameters['min_samples_leaf'])))
    max_depths = cv_results['param_max_depth'].reshape(scores.shape).data.astype(int)
    min_samples_leafs = cv_results['param_min_samples_leaf'].reshape(scores.shape).data.astype(int)
    
    # Plot result
    ax.plot_wireframe(max_depths, min_samples_leafs, scores)
    ax.view_init(20, 220)
    ax.set_xlabel('Maximum Depth')
    ax.set_ylabel('Minimum Samples Leaf')
    ax.set_zlabel('Accuracy')
    ax.set_title(title)


fig = plt.figure(figsize=(15,7))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
plot_scores(ax1, 'Train Accuracy', search, hyperparameters, 'mean_train_score')
plot_scores(ax2, 'Validation Accuracy', search, hyperparameters, 'mean_test_score')

# Now, we would want to train on a type of random forest (RandomForest416).
class RandomForest416: 
    """
    This class implements the common sklearn model interface (has a fit and predict function).
    
    A random forest is a collection of decision trees that are trained on random subsets of the 
    dataset. When predicting the value for an example, takes a majority vote from the trees.
    """
    
    def __init__(self, num_trees, max_depth=1):
        """
        Constructs a RandomForest416 that uses the given number of trees, each with a 
        max depth of max_depth.
        """
        self._trees = [
            DecisionTreeClassifier(max_depth=max_depth) 
            for i in range(num_trees)
        ]
        
    def fit(self, X, y):
        """
        Takes an input dataset X and a series of targets y and trains the RandomForest416.
        
        Each tree will be trained on a random sample of the data that samples the examples
        uniformly at random (with replacement). Each random dataset will have the same number
        of examples as the original dataset, but some examples may be missing or appear more 
        than once due to the random sampling with replacement.
        """    
        # Q7
        # TODO 

        for tree in self._trees:
            rand_indices = np.random.randint(0, len(X), len(X))

            X_sample = X.iloc[rand_indices]
            y_sample = y.iloc[rand_indices]

            tree.fit(X_sample, y_sample)
           
    def predict(self, X):
        """
        Takes an input dataset X and returns the predictions for each example in X.
        """
        # Builds up a 2d array with n rows and T columns
        # where n is the number of points to classify and T is the number of trees
        predictions = np.zeros((len(X), len(self._trees)))
        for i, tree in enumerate(self._trees):
            # Make predictions using the current tree
            preds = tree.predict(X)
            
            # Store those predictions in ith column of the 2d array
            predictions[:, i] = preds
            
        # For each row of predictions, find the most frequent label (axis=1 means across columns)
        return scipy.stats.mode(predictions, axis=1, keepdims=False).mode 


# Compute the training and validation accuracies on the RandomForest416 model below.
# Random Forest fitting
rf = RandomForest416(num_trees = 2, max_depth = 1)
rf.fit(train_data[features], train_data[target])


# Random Forest training data accuracy
rf_pred_train = rf.predict(train_data[features])
print(rf_pred_train)

rf_train_accuracy = accuracy_score(train_data[target], rf_pred_train)
print(rf_train_accuracy)


# Random Forest validation data accuracy
rf_pred_val = rf.predict(validation_data[features])
print(rf_pred_train)

rf_validation_accuracy = accuracy_score(validation_data[target], rf_pred_val)
print(rf_validation_accuracy)

# Let's compare the decision tree and RandomForest416 models on their training and validation accuracy metric.
# First calculate the accuracies for each depth
depths = list(range(1, 26, 2))
dt_accuracies = []
rf_accuracies = []

for i in depths:
    print(f'Depth {i}')

    # Train and evaluate a Decision Tree Classifier with given max_depth
    tree = DecisionTreeClassifier(max_depth=i)
    tree.fit(train_data[features], train_data[target])
    dt_accuracies.append((
        accuracy_score(tree.predict(train_data[features]), train_data[target]),
        accuracy_score(tree.predict(validation_data[features]), validation_data[target])
    ))
    
    # Train and evaluate our RandomForest classifier with given max_depth 
    rf = RandomForest416(15, max_depth=i)
    rf.fit(train_data[features], train_data[target])
    rf_accuracies.append((     
        accuracy_score(rf.predict(train_data[features]), train_data[target]),
        accuracy_score(rf.predict(validation_data[features]), validation_data[target])
    ))
    
# Then plot the scores
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot training accuracies
axs[0].plot(depths, [acc[0] for acc in dt_accuracies], label='DecisionTree')
axs[0].plot(depths, [acc[0] for acc in rf_accuracies], label='RandomForest416')

# Plot validation accuracies
axs[1].plot(depths, [acc[1] for acc in dt_accuracies], label='DecisionTree')
axs[1].plot(depths, [acc[1] for acc in rf_accuracies], label='RandomForest416')

# Customize plots
axs[0].set_title('Train Data')
axs[1].set_title('Validation Data')
for ax in axs:
    ax.legend()
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Accuracy')