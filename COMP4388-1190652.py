"""" Project 2 for Machine Learning Course by Dr. Radi Jarrar, 
     Classification task for Weather Dataset 
     Sondos Aabed """

#Import necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import learning_curve

""" Read the Weather dataset file """
def read_file(name):
    dataframe = pd.read_excel(name) # Load the excel file into a data Structure
    print("Observations: ", len(dataframe))  # check the size of obsevations
    print("Features: ", dataframe.columns.size) # check number of features
    return dataframe

# high cardinality of date needs to be handeled
def handle_date(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Year'].head()
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Month'].head()
    dataframe['Day'] = dataframe['Date'].dt.month
    dataframe['Day'].head()
    dataframe.drop('Date', axis=1, inplace = True)

""" Chapter one: Eploratory Data Analysis (EDA) """
    #####
    #    Features Overview
    ####

# draw histograms for quantitative features
def histograms(dataframe):
    # for these attributes use bin = 15
    # Cloud 3 pm cooulmn
    figure, axes = plt.subplots(figsize=(20, 20))
    dataframe['Cloud3pm'].hist(ax=axes, bins=15, color='orange')
    axes.set_xlabel('Cloud 3 pm')
    axes.set_ylabel("Count")
    # Cloud coulmn
    figure1, axes1 = plt.subplots(figsize=(20, 20))
    dataframe['Cloud9am'].hist(ax=axes1, bins=15, color='orange')
    axes1.set_xlabel('Cloud 9 am')
    axes1.set_ylabel("Count")
    # Rain fall coulmn 
    figure3, axes3 = plt.subplots(figsize=(20, 20))
    dataframe['Rainfall'].hist(ax=axes3, bins=15, color='orange')
    axes3.set_xlabel('Rain Fall')
    axes3.set_ylabel("Count")
    # Calculate the bin size based on the square root of the count of that feature
    for col in dataframe:
        if dataframe[col].dtype != object: # Check it's quanitative
            figure2, axes2 = plt.subplots(figsize=(20, 20))
            n_bins = int(np.sqrt(len(dataframe))) # set bin size the square root of count
            dataframe[col].hist(ax=axes2, bins=n_bins, color='orange')
            axes2.set_xlabel(col)
            axes2.set_ylabel('Count')

# draw pie Chart for qualitative features
def pies(dataframe):
    for col in dataframe:
        if dataframe[col].dtype == object: # Check it's qualitative
            figure, axes = plt.subplots(figsize=(20, 20))
            dataframe[col].value_counts().plot(kind='pie', ax=axes, autopct='%1.1f%%')
            axes.set_title(col)

    #####
    #    Data Cleansing
    ####

# draw box plots to detect outliers
def boxPlot(dataframe):
    for col in dataframe:
        if (dataframe[col].dtype != object and dataframe[col].dtype != '<M8[ns]') :
            # Plot the box plot for a specific column
            figure, axes = plt.subplots(figsize=(20, 20))
            sns.boxplot(dataframe[col],ax=axes)
            axes.set_title(col)

# draw scatter plot to detect outliers
def scatterPlot(dataframe,col):
    figure, axes = plt.subplots(figsize=(20, 20))
    sns.scatterplot(dataframe[col],ax=axes)
    axes.set_title(col)

# use KNN to impute missing values
def impute_KNN(df):
    imputer = KNNImputer(n_neighbors=10)
    imputed = imputer.fit_transform(df)
    df_imputed = pd.DataFrame(imputed, columns=df.columns)
    return df_imputed

# this method returns the lower and the upper bound of a feature
def find_outliers(dataframe, col):
    q1 = dataframe[col].quantile(0.25)
    q3 =dataframe[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5*iqr)
    upper_bound = q3 + (1.5*iqr)
    return [lower_bound,upper_bound]

# this method imputes qualitative missing data and convert them to numerical
def impute_and_convert(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
            # impute missing values with the mode of the column
            dataframe[col].fillna(dataframe[col].mode()[0], inplace=True)
            # convert column to numerical representation
            dataframe.loc[:, col] = dataframe[col].astype('category').cat.codes
    return dataframe
    
def clean_data(dataframe):
    # remove duplicate rows
    dataframe=dataframe.drop_duplicates()

    # convert non-numerical values to numerical
    dataframe= impute_and_convert(dataframe)

    # fill in the missing data with KNN
    dataframe= impute_KNN(dataframe)

    # these features were detected to have outliers using EDA
    outliers=['Rainfall','WindSpeed3pm','WindGustSpeed','WindSpeed9am','Humidity9am','Pressure3pm','Pressure9am','Temp3pm','Temp9am','MaxTemp']
    # handle outliers
    for col in outliers:
        outliers_bounds = find_outliers(dataframe,col)
        dataframe[col]=np.where(dataframe[col]>outliers_bounds[1],outliers_bounds[1],np.where(dataframe[col]<outliers_bounds[0],outliers_bounds[0],dataframe[col]))
    # return cleansed datadrame
    return dataframe

# feature scaling using minmax
def min_max(dataframe):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(dataframe)
    return pd.DataFrame(scaled, columns=dataframe.columns)
        
    #####
    #   Data Coralaation
    ####

# draw heat map to check data corelation
def heatMap(dataframe):
    figure12, axes12= plt.subplots()
    sns.set(rc = {'figure.figsize':(16,8)})
    sns.heatmap(dataframe.corr(), ax=axes12,annot = True,fmt='.2g',cmap= 'crest',linewidth=.5)

# based in the correlation between features
def select_features(dataframe):
    dataframe= dataframe[['Month', 'Location','MinTemp','MaxTemp','Humidity9am','Humidity3pm','WindGustSpeed','WindDir9am','WindDir3pm','Cloud9am','Cloud3pm','RainTomorrow','RainToday']]
    return dataframe

""" Chapter two: Classification Algorithms """
# this function is used to return performance measure for the model
# three models will be passed through this function are:
""" 
SVC()
MLPClassifier()
LogisticRegression()
"""
def evaluate_model(model, x_test, y_test, X_train, Y_train):
    if isinstance(model, SVC):
        model = SVC(probability=True)
    
    if isinstance(model, MLPClassifier):
        model = MLPClassifier(max_iter=1000)

    # Calculate accuracy measurments
    model.fit(X_train, Y_train)
    y_pred = model.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    recall = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate auc
    y_pred_proba = model.predict_proba(x_test)[::,1]
    auc = metrics.roc_auc_score(y_test, y_pred_proba)

    # confussion matrix
    confusionMatrix = metrics.confusion_matrix(y_test, y_pred)
    return [ accuracy, precision, recall, f1, auc, confusionMatrix ]

# cross validation and plot learning curve
def cv_learnCurve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    figure, axes = plt.subplots()
    axes.plot(train_sizes, train_scores.mean(axis=1), label='Training score')
    axes.plot(train_sizes, test_scores.mean(axis=1), label='Test score')
    axes.set_xlabel('Training examples')
    axes.set_ylabel('Score')
    axes.legend(loc='best')

""" Main function """
if __name__ == "__main__":
    """ Read data file """
    dataframe = read_file('WeatherData.xls')
    # saves the output of describe method into a cvs file
    round(dataframe.describe(),2).to_csv("quant.csv") 
    dataframe.describe(include=['object']).to_csv("qualit.csv") 
    handle_date(dataframe)

    """ Features Overview """
    histograms(dataframe)
    pies(dataframe)
    
    """ Data cleansing """
    # detect outliers
    boxPlot(dataframe) 
    scatterPlot(dataframe,'Rainfall')
    
    # clean dataframe (missing data and outliers handling)
    dataframe=clean_data(dataframe)
    round(dataframe.describe(),2).to_csv("afterCleansing.csv") 
    
    """ Data preprocessing """
    # perform feature scaling using min max algorithm for a new dataframe 
    dataframeScaled = min_max(dataframe)
    round(dataframe.describe(),2).to_csv("afterScaling.csv") 

    """ Data Corellation """
    heatMap(dataframeScaled)

    """ Features Selection """
    dataframe = select_features(dataframe)
    dataframeScaled= select_features(dataframeScaled)
     
    """ Classification """
    # split the data and drop the target feature
    X = dataframeScaled.drop(['RainTomorrow'], axis=1)
    Y = dataframeScaled['RainTomorrow']
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    
        ### 
        # Evaluation returned array content
        # accuracy, precision, recall, f1, auc, confusionMatrix
        ### 

    # Train & evaluate logetsic regression model and get predictions
    lr_results = evaluate_model(LogisticRegression(), X_test, y_test, X_train, y_train)
    print("Logestic regression Model Results: ", lr_results)
    cv_learnCurve(LogisticRegression(),X, Y)
    
    # Train & evaluate SVM model and get predictions
    svm_results = evaluate_model(SVC(), X_test, y_test, X_train, y_train)
    print("SVM Model Results: ", svm_results)
    cv_learnCurve(SVC(),X, Y)

    # Train and evaluate ANN model and get predictions
    ann_results = evaluate_model(MLPClassifier(), X_test, y_test, X_train, y_train)
    print("ANN Model Results: ", ann_results)
    cv_learnCurve(MLPClassifier(),X, Y)

    plt.show()
