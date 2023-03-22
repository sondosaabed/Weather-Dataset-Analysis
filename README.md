# Weather-Dataset-Analysis:
This project was created as part of Machine Learning Course at BZU. After completing the analysis of the Weather Data provided, the results can be found in the document provided. Based on the analysis conclusions were found that will better decision making concerning the weather if it will rain tomorrow or not. 

## <strong>Features overview:</strong>
Histograms that shows the features distribution: </br>

![image](https://user-images.githubusercontent.com/65151701/218209753-2d3fe781-c02d-46d3-9842-b0b008b9a7b5.png)

Outliers detection using box plots: </br>

![image](https://user-images.githubusercontent.com/65151701/218209869-51715e7b-ec26-435d-b47a-3af3b368226f.png)

Showing Data corelation using Heat map: </br>

![image](https://user-images.githubusercontent.com/65151701/218210013-32d7ac75-138f-4341-85c3-918bf5c93787.png)

## <strong>Models trained </strong></br>
Logestic reggression: </br>

![image](https://user-images.githubusercontent.com/65151701/218210296-3ea7d3cf-2ace-4ec0-ba70-7134ba956308.png)

Support Vector Machine: </br>

![image](https://user-images.githubusercontent.com/65151701/218210441-43e07baa-8eaa-43d1-a029-38a06d1b6233.png)

Artificial Neural Network: </br>

![image](https://user-images.githubusercontent.com/65151701/218210674-b61a5428-6f2c-4cb2-9929-ee0b72ebe330.png)

# Conclusions
<p>The analysis of the weather dataset revealed the presence of missing values and outliers that were effectively handled using KNN imputer and Capping and Flooring. It also showed that the ranges of some features may dominate the others in respect to their contribution to the classification task so feature scaling had to be performed. </p>
<p>Multivariate analysis was performed to determine the correlation between features and the target, leading to the removal of features with low positive correlation and highly correlated features to avoid redundancy.</p>
<p>After evaluating the performance of three different classification algorithms (LR, ANN and SVM), the ANN classifier was found to be the best performer with the highest ROC/AUC score and precision.  For that reason the selected algorithm will be ANN to predict whether it will rain tomorrow or not. </p>
<p>These results can be valuable for Al-Bireh municipality in making informed decisions about weather predictions, such as allocating resources for potential rain-related events or planning outdoor activities based on the predicted weather. By using the best performing classifier, the municipality can have more confidence in its weather predictions and respond more effectively to potential weather-related challenges.</p>
