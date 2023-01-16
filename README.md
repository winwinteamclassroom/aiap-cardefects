# a. Full name (as in NRIC) and email address (stated in your application form)

    Name: Chong Wei Sheng
    email address: ws.chong.sg@gmail.com

# b. Overview of the submitted folder and the folder structure

    \aiap13-ChogWeiSheng-024
        └──\src
        │   └── main.py
        ├── eda.ipynb
        ├── README.md
        ├── requirements.txt
        ├── run.sh

# c. Instructions for executing the pipeline and modifying any parameters

    source ./run.sh    

# d. Description of logical steps/flow of the pipeline

    The pipeline first clean the dataset, the split the data into train and test data, train the model using train data, then evaluate the model using test data.

# e. EDA overview

    First, I remove duplicated rows, remove small number of rows with wrong factory location, fill up empty data. Then I change categorical data into numerical features. I also aggregate the failure to be predicted from 5 indivisual failures to check if any of the failures occurs. 

    I check the correlation of the numerical features with the aggregated failure to find out the dominant features for training.

# f. Described how the features in the dataset are processed (summarized in a table)

# g. Explanation of your choice of models for each machine learning task

    This is a binary classification problem. I use Support Vector Machine and logistic regression.

# h. Evaluation of the models developed

    I check the accuracy and confusion matrix. I reduce the number of features to reduce complexity of the model.

# i. Other considerations for deploying the models developed
