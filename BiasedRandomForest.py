import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
import math
import random
from math import sqrt
from numpy import mean
from scipy.stats import sem, t
%matplotlib inline

plt.style.use('ggplot')
default_blue_hex = '#1f77b4'

import warnings
warnings.filterwarnings('ignore')


def train_test_split(df, test_size):
    """
    This function splits the data into train and test set
    df : a pandas datframe with data that needs to be split
    test_size: fraction of input data that will be used for creating test data set
    returns: two dataframes; training dataframe and testing dataframe respectively
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


def determine_type_of_feature(df):
    """
    df: a pandas dataframe with mix of different types of variables
    returns: list with datatype for each feature/column in the input dataframe
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types
    

def check_purity(data):
    """
    This function is responsible for checking if a split is pure
    data: an array containing data including class labels
    returns: Boolean value indicating purity of a split
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    #selecting target variable and it's uniques values
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    #if all the records in a dataset belong to same class then the datset is pure
    if len(unique_classes) == 1:
        return True
    else:
        return False

    
def classify_data(data):
    """
    This function is responsible for assigning class label based on majority vote
    data: a dataset with class label feature
    returns: the class label, calculated based on majority vote
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    #selecting class labels
    label_column = data[:, -1]
    #calculating record counts for each class label
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    #picking class label with more records assigned to it
    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


def get_potential_splits(data, random_subspace):
    """
    data: a dataset with predictors
    random_subspace: number of predcitors to randomly select
    returns: dictionary with predictor index as the key and it's unique values as the value
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))    # excluding the last column which is the label
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    
    for column_index in column_indices:          
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


def calculate_entropy(data):
    """
    This function calculates the entropy for a given dataset. This function assumes last column to be the class
    label.
    data: a dataset with class label
    returns: entropy for the input dataset
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy


def calculate_overall_entropy(data_below, data_above):
    """
    This function calculates overall entropy for a split. It takes the weighted sum of data above and below the 
    split value as overall entropy. This score is used to pick the best combination of the predictor and a value 
    used for splitting the data.
    data_below: a dataset with records below the predictor split value
    data_above: a dataset with records above the predictor split value
    returns: weighted sum of both the input datasets' entropy
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """

    #calculating the size of the dataset
    n = len(data_below) + len(data_above)
    #calculating data proportion
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    #weighted cross entropy calculation
    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy


def determine_best_split(data, potential_splits):
    """
    data: a dataset with predictors
    potential_splits: list of predictors index identified in random selection process
    returns: combination of best split predictor and value
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    # initializing cross entropy
    overall_entropy = 9999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            #splitting data into two chunks
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            #calculating cross entropy for the split
            current_overall_entropy = calculate_overall_entropy(data_below, data_above)
            #picking the best split predictor and value
            if current_overall_entropy <= overall_entropy:
                overall_entropy = current_overall_entropy
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value


def split_data(data, split_column, split_value):
    """
    This function splits the input data on provided split value of the split column
    data: a pandas dataframe with predictors and their values
    split_column: column name, based on which the datset will be split
    split_value: value of split column on which the dataset will be divided
    returns: two datset; a dataset containing split column value below the split 
             value and a dataset with values above the split value respectively
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    split_column_values = data[:, split_column]

    # identifying the predictors data type
    type_of_feature = FEATURE_TYPES[split_column]
    
    # if the data is continuous, then two datasets contains records with values below
    # and above the split value. Whereas in case of categorical variable, one dataset
    # consists of records with split column value same as split value and others with value
    # not similar to split value
    
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above


def decision_tree_algorithm(df, counter=0, min_samples=2, max_depth=5, random_subspace=None):
    """
    This function build decision trees.
    df: a pandas dataframe with predictors and dependent variable. The functions expects the 
        last column of the input dataset to be target variable
    counter: this is the variable which keeps track of the depth of the decision tree
    min_samples: this indicated minimum size of dataset to split on
    max_depth: this variables indicated the maximum depth of the decision tree
    random_subspace: this variable contains information about the number of predictors to randomly select
                     out of all the available to identify the best split column and value
    returns: decision tree
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data, random_subspace)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, random_subspace)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, random_subspace)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    

def predict_example(example, tree):
    """
    This function calculates decision tree prediction for provided data sample
    example: a data sample for which the prediction needs to be made
    tree: a decision tree
    returns: class label for the provided data sample
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    question = list(tree.keys())[0]
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)
    

def decision_tree_predictions(test_df, tree):
    """
    This function is entry point for predicting class label from a decision tree for a dataset
    test_df: a pandas dataframe with predictor values
    tree: a decision tree
    returns: a dataset with predictions for the input dataset
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    predictions = test_df.apply(predict_example, args=(tree,), axis=1)
    return predictions


def bootstrapping(train_df, n_bootstrap):
    """
    This function is responsible for bootstrapping the training data.
    train_df: a pandas dataframe with predictors
    n_bootstrap: size of the bootstrap sample
    returns: a pandas dataframe with randomly selected samples
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped


def random_forest_algorithm(train_df, n_trees, bootstrap_frac, features_frac, dt_max_depth):
    """
    This function build the forest of trees. The purpose of this function is to train a 
    random forest classifier on the input dataset
    train_df: training dataset
    n_trees: number of trees to build
    bootstrp_frac: size of bootstrap sample in fraction
    feature_frac: fraction of predictors to select while creating each node under a tree
    dt_max_depth: maximum depth of the trees
    returns: list(forest) of decision trees
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    #calculating number of features and samples for bootstrapping
    n_features = int(features_frac * (len(train_df.columns)-1))
    n_bootstrap = int(bootstrap_frac * len(train_df))
    forest = []
    #building trees
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    
    return forest


def random_forest_predictions(test_df, forest):
    """
    This function predicts class label for input dataset. The technique used to calculate the
    prediction is majority voting.
    test_df: a pandas dataframe with predictors to predict on
    forest: list of decision trees / random forest classifier
    returns: predictions for input dataset
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/SebastianMantey/Random-Forest-from-Scratch
    """
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]
    return random_forest_predictions


def stratified_split(df, n_folds):
    """
    This function creates stratified fold of the input data. The stratification is performed on the class 
    label column which is presumed to be the last column in the dataset
    df: a pandas dataframe to sample data from
    n_folds: number of chunks to divide the input dataset into
    returns: list of dataset 
    ------------------------------------------------------------------------------------------------------
    Reference:
    https://github.com/khataei/Cross-validation-from-scratch
    """

    rows = df.shape[0]  # number of total rows
    # maximum number of rows in each fold
    fold_row = math.ceil(rows / n_folds)
    fold_column = df.shape[1]  # number of features
    folds = np.empty((n_folds, fold_row, fold_column)
                     )  # creating empty folds,
    # set the fold toNan so eliminating them would be easy by pd.dropna
    folds[:] = np.nan

    # if need to shuffle use this:
    # suffle them with sklearn
    # from sklearn import utils
    # initial=utils.shuffle(initial)

    # the one with 0 in the last column are class A
    class_A_data = df[df.iloc[:, fold_column - 1] == 0]
    class_B_data = df[df.iloc[:, fold_column - 1] == 1]

    class_A_data = class_A_data.values  # convert pandas Data frame to numpy array
    class_B_data = class_B_data.values
    class_A_size = class_A_data.shape[0]  # sizeA is the number of instances in class A

    for i in range(class_A_size):  # asigning almost equal number of class A and B to folds
        folds[i % n_folds, i // n_folds, :] = class_A_data[i, :]

    for i in range(class_A_size, rows, 1):  # we start from sizeA to continue with the next fold, not necceserily the first one
        folds[i % n_folds, i // n_folds, :] = class_B_data[i - class_A_size, :]
    
    # Untill here we read the data, devided into k fold, each fold has same
    # number of class A and B
    return folds


def get_stratified_folds(train_df, n_folds):
    """
    This functions is entry point to create stratified samples from the input data. This function is also 
    responsible for label encoding the folder index.
    train_df: a pandas dataframe to sample data from
    n_folds: number of chunks to divide the input dataset into
    returns: a pandas dataframe with new columns "fold" highlighting the fold number a sample belongs to
    """
    stratified_folds = stratified_split_fold(train_df, n_folds)
    cv_df_list = []
    col_names = train_df.columns
    for fold in range(n_folds):
        tmp_df = pd.DataFrame(stratified_folds[fold])
        tmp_df.columns = col_names
        tmp_df['fold'] = fold
        cv_df_list.append(tmp_df)
    cv_df = pd.concat(cv_df_list)
    cv_df.dropna(thresh= cv_df.shape[1], inplace=True)
    return cv_df


def random_forest_probability_predictions(test_df, forest):
    """
    This function predicts probability for positive class label on the input dataset. The technique
    used to calculate the probability is: 
        = number of trees predicting positive class / total number of trees
    test_df: a pandas dataframe with predictors to predict on
    forest: list of decision trees / random forest classifier
    returns: probability predictions for input dataset
    """
    df_predictions = {}
    for i in range(len(forest)):
        column_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(test_df, tree=forest[i])
        df_predictions[column_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.sum(axis=1)/df_predictions.shape[1]
    return random_forest_predictions


def normalize_data(data):
    """
    This function is an implementation of MinMax Scaler
    data: a dataset
    returns: normalized dataset
    """
    return (data - data.min())/(data.max() - data.min())


def get_majority_class_nearest_neighbours(majority_data, minority_data, n):
    """
    This function identifies majority class nearest neighbors of minority class
    majority_data: predcitor values for records that belong to majority class
    minority_data: predcitor values for records that belong to minority class
    n: number of nearest neighbors to identify
    returns: list of unique majority class records that appeared in top n neighbors
             list of any of the minority class record
    """
    overrepresented_class_neighbors = set()
    # Normalizing the input dataset. Since euclidean distance is being calculated it is 
    # good to normalize the data to same scale so the distance calculation is not affected 
    # by the variation in scale of variables
    normalized_majority_data = normalize_data(majority_data)
    normalized_minority_data = normalize_data(minority_data)
    for row in normalized_minority_data.iloc[:, :-1].values:
        #identifying the top n neighbors
        overrepresented_class_neighbors.update((np.linalg.norm(row - normalized_majority_data.iloc[:, :-1], 
                                                               axis=1)).argsort()[:n])
    return majority_data.iloc[list(overrepresented_class_neighbors)]


def identify_class_labels(data):
    """
    This function identifies the class labels
    data: a dataset with target variable
    returns: majority label and minority label
    """
    label_column = data.iloc[:, -1]
    labels, counts = np.unique(label_column, return_counts=True)
    majority_label = labels[counts.argmax()]
    minority_label = labels[counts.argmin()]
    return majority_label, minority_label


def biased_random_forest_algorithm(train_df, K, p, s, bootstrap_frac=0.8, feature_frac=0.7, max_depth=3):
    """
    This function implements the Biased RAndom Forest algorithm as described in the paper referenced below
    train_df: a pandas dataframe to train the algorithm on
    K: number of nearest neighbors to identify while creating critical dataset
    p: fraction of trees to build with critical data
    s: combined number of trees to build
    bootstrp_frac: size of bootstrap sample in fraction
    feature_frac: fraction of predictors to select while creating each node under a tree
    max_depth: maximum depth of the trees
    returns: combined list of trees
    """
    # identifying majority and minority class label in impbalanced dataset
    majority_label, minority_label = identify_class_labels(train_df)
    
    majority_label_data = train_df[train_df.iloc[:, -1] == majority_label]
    minority_label_data = train_df[train_df.iloc[:, -1] == minority_label]
    
    # creating critical dataset. This dataset is created by taking all the minority class samples and 
    # only the majority class samples that appear in top n nearest neighbors to the minority class records
    critical_dataset = get_majority_class_nearest_neighbours(majority_label_data, minority_label_data, K)
    balanced_df = pd.concat([minority_label_data, critical_dataset])
    
    # converting trees fraction to number of trees
    n_trees = int(p*s)
    # building random forest with original dataset
    forest = random_forest_algorithm(train_df, n_trees=(s-n_trees), bootstrap_frac=bootstrap_frac, 
                                     features_frac=feature_frac, dt_max_depth=max_depth)
    # building random forest with critical dataset
    balanced_forest = random_forest_algorithm(balanced_df, n_trees=n_trees, bootstrap_frac=bootstrap_frac, 
                                              features_frac=feature_frac, dt_max_depth=max_depth)
    # combining trees
    forest.extend(balanced_forest)
    return forest

def set_seed(random_seed):
    """
    This function is responsible to set random seed for reproducability purpose
    random_seed: value to which the random seed will be set
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

def evaluate_braf_algorithm(df, K= 10, p = 0.5, s=100, n_folds=10, random_seed=123):
    """
    This function performs multiple steps, including: 
        - spliting inut data into train and test set,
        - performing cross validation on training dataset
        - building BRAF model on training dataset
        - predicting outcome for test dataset
        - displaying classification report
    df: a pandas dataframe to evaluate BRAF algorithm
    K: number of nearest neighbors to identify while creating critical dataset
    p: fraction of trees to build with critical data
    s: combined number of trees to build
    n_folds: number of chunks to create for cross validation
    random_seed: value to which the random seed will be set
    returns: two datasets; one with cross validation predictions and other with test data predictions
    """
    # assigning the random seed value
    set_seed(random_seed)
    label_column_name = df.columns[-1]
    
    # spliting input data
    train_df, test_df = train_test_split(df, test_size=0.3)
    cv_df = get_stratified_folds(train_df, n_folds)
    
    prediction_df = pd.DataFrame()
    # performing cross validation on the training dataset
    for fold in range(n_folds):
        # identifying train and test dataset
        cv_train_df = cv_df[cv_df.fold != fold].drop(['fold'], axis=1)
        cv_test_df = cv_df[cv_df.fold == fold].drop(['fold'], axis=1)
        
        # training BRAF algorithm
        forest = biased_random_forest_algorithm(cv_train_df, K, p, s)
        # predicting probability for test dataset
        cv_test_df['pred_prob'] = random_forest_probability_predictions(cv_test_df, forest)
        cv_test_df['pred'] = cv_test_df['pred_prob'].apply(lambda x: 1 if x>=0.5 else 0)
        prediction_df = pd.concat([prediction_df, cv_test_df])
    print ("---------------------------Cross validation result---------------------------")
    # generating performance report for cross validation results
    recall, precision, roc_auc, pr_auc = calculate_classification_metrics(prediction_df[label_column_name],
                                                                          prediction_df['pred_prob'], plot_title_label='cross validation')
    print("Recall score       :", recall)
    print("Precision score    :", precision)
    print("Confusion matrix   :\n", prediction_df.groupby([label_column_name, 'pred']).size().unstack())
    
    # training BRAF algorithm on training dataset
    forest = biased_random_forest_algorithm(train_df, K, p, s)
    # predicting probabilities for test dataset
    test_df['pred_prob'] = random_forest_probability_predictions(test_df, forest)
    test_df['pred'] = test_df['pred_prob'].apply(lambda x: 1 if x>=0.5 else 0)
    print ("-------------------------------Test set result-------------------------------")
    # generating performance report for test dataset predictions
    recall, precision, roc_auc, pr_auc = calculate_classification_metrics(test_df[label_column_name], 
                                                                          test_df['pred_prob'], plot_title_label='Test')
    print("Recall score       :", recall)
    print("Precision score    :", precision)
    print("Confusion matrix   :\n", test_df.groupby([label_column_name, 'pred']).size().unstack())
    return prediction_df[[label_column_name, 'pred', 'pred_prob']], test_df[[label_column_name, 'pred', 'pred_prob']]


def auc(x, y):
    """
    This function calculates the area under the curve
    x: x axis values
    y: y axis values
    returns: area under curve 
    """
    auc = np.trapz(x, y)
    return auc
    

def identify_true_labels(y_true, y_pred):
    """
    This function identifies the count of samples with positive class label. This function assumes 
    the target valirable is binary.
    y_true: observed class labels
    y_pred: predicted class labels
    returns:
    """
    y_true_ind = []
    for i, j in enumerate(y_true == 1):
        if j:
            y_true_ind.append(i)

    y_pred_ind = []
    for i, j in enumerate(y_pred == 1):
        if j:
            y_pred_ind.append(i)
    return y_true_ind, y_pred_ind

def recall_score(y_true, y_pred):
    """
    This function calculates recall score for input observed and predicted labels
    y_true: observed class labels
    y_pred: predicted class labels
    returns: recall score
    """
    y_true_ind, y_pred_ind = identify_true_labels(y_true, y_pred)
    return np.round(len(np.intersect1d(y_pred_ind, y_true_ind))/len(y_true_ind), 3)


def precision_score(y_true, y_pred):
    """
    This function calculates precision score for input observed and predicted labels
    y_true: observed class labels
    y_pred: predicted class labels
    returns: precision score
    """
    y_true_ind, y_pred_ind = identify_true_labels(y_true, y_pred)
    return np.round(len(np.intersect1d(y_pred_ind, y_true_ind))/len(y_pred_ind), 3)


def plot_auc(metrics_df, plot_title_label=None):
    """
    This functioons visualizes and stores the PR curve and ROC curve
    metrics_df: a pandas dataframe with recall, precision and false positive rate calculated for 
                all possible probability threshold values
    plot_title_label: label of plot titles
    """
    # calculating area under ROC curve
    roc_auc = auc(metrics_df['recall'], metrics_df['false_positive_rate'])
    # calculating area under PR curve
    pr_auc = auc(metrics_df['precision'], metrics_df['recall'])
    
    #visualizing PR curve
    plt.plot(np.append(metrics_df['precision'], [0]), np.append([0], metrics_df['recall']))
    plt.xlabel('precision')
    plt.ylabel('recall')
    if plot_title_label:
        title = plot_title_label + "- Precision recall curve (Score:" + str(np.round(pr_auc, 4)*100) + ")"
    else:
        title = "Precision recall curve (Score:" + str(np.round(pr_auc, 4)*100) + ")"
    plt.title(title)
    if plot_title_label:
        plt.savefig(plot_title_label+' PR curve.png')
    else:
        plt.savefig('PR curve.png')
    plt.show()
    
        
    #visualizing ROC curve
    plt.plot(metrics_df['false_positive_rate'], metrics_df['recall'])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if plot_title_label:
        title = plot_title_label + "- ROC curve (Score:" + str(np.round(roc_auc, 4)*100) + ")"
    else:
        title = "ROC curve (Score:" + str(np.round(roc_auc, 4)*100) + ")"
    plt.title(title)
    if plot_title_label:
        plt.savefig(plot_title_label+' ROC curve.png')
    else:
        plt.savefig('ROC curve.png')
    plt.show()
    return roc_auc, pr_auc

def calculate_classification_metrics(y_true, y_prob, plot_title_label=None):
    """
    This function generates classification report
    y_true: observed class labels
    y_prob: predicted probabilities
    plot_title_label: label of plot titles
    returns: recall score, precision score, area under ROC curve, area under PR curve
    """
    # combining observed labels and predicted probabilites
    metrics_df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    
    #calculating predicted class labels
    metrics_df['y_pred'] = metrics_df['y_prob'].apply(lambda x: 1 if x>=0.5 else 0)
    
    #calculating recall and precision score
    recall = recall_score(metrics_df['y_true'], metrics_df['y_pred'])
    precision = precision_score(metrics_df['y_true'], metrics_df['y_pred'])
    
    # identifying number of positive & negative class labels observed and total number of records 
    # under each probability threshold
    metrics_df = metrics_df.groupby('y_prob').agg({'y_true': {'positive_class_count': sum, 'record_count' : np.size}})
    metrics_df.columns = metrics_df.columns.droplevel()
    metrics_df = metrics_df.sort_index(ascending=False).cumsum()
    metrics_df['negative_class_count'] = metrics_df['record_count'] - metrics_df['positive_class_count']
    
    #calculating precision, recall and false positive rate at each probability threshold
    metrics_df['precision'] = metrics_df['positive_class_count'] / (metrics_df['positive_class_count'] + metrics_df['negative_class_count'])
    metrics_df['precision'] = metrics_df['precision'].fillna(0)
    metrics_df['recall'] = metrics_df['positive_class_count'] / metrics_df['positive_class_count'].max()
    metrics_df['false_positive_rate'] = metrics_df['negative_class_count'] / metrics_df['negative_class_count'].max()
    roc_auc, pr_auc = plot_auc(metrics_df, plot_title_label)
    return recall, precision, roc_auc, pr_auc
