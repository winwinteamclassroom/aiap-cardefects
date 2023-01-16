"""
reference:
    https://www.kaggle.com/code/surajjha101/heart-failure-prediction-svm-and-ann
"""
import sqlite3
import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn import svm, linear_model

failure_list = ["Failure A","Failure B", "Failure C", "Failure D", "Failure E"]
numerical_feature = ['RPM', 
    # 'Failure A', 'Failure B', 'Failure C', 'Failure D', 'Failure E', 
    'Mod_yy', 'Mod_num', 'Tem_celc', 'Color_num', 'Fac_ctry_num', 'Fac_city_num', 
    'Usage_num', 'Membership_num', 'Total_fail']

def remove_wrong_city(df, wrong_city = ["Seng Kang, China", "Newton, China", "Bedok, Germany"]):
    for city in wrong_city:
        print(city)
        df = df[df.Factory.str.match(city)==False]
    
    return df

def categorical_to_numerical(df):
    df["Mod_yy"]=df.Model.apply(lambda x:int(x.split(",")[-1][-2:]))
    df["Mod_num"]=df.Model.apply(lambda x: int(x.split(",")[0][-1]))
    df["Fac_ctry"]=df.Factory.apply(lambda x:x.split(",")[-1])
    df["Fac_city"]=df.Factory.apply(lambda x:x.split(",")[0])
    df["Tem_celc"]=df.Temperature.apply(
        lambda x: float(x.split(" ")[0]) if x.split(" ")[-1]=="°C" else (float(x.split(" ")[0]) - 32)*(5/9))

    cat_col = ["Color",	"Fac_ctry", "Fac_city"]
    cat2num_dict = {}
    for col in cat_col:
        cat2num_dict[col] = sorted(list(df[col].unique()))    

    cat2num_dict["Usage"] = ["Low", "Medium", "High"]
    cat2num_dict["Membership"] = ["Premium", "Normal", "None", "Empty"]

    for col, cat in cat2num_dict.items():
        new_col = col+"_num"
        df[new_col] = df[col].apply(lambda x: cat.index(x))
        print(f"New numerical column {new_col} and its matched value ")
        for c in cat:
            print(f"\t{cat.index(c)}:{c}")    
            
    return df, cat2num_dict

def preprocessing(df):
    df.drop_duplicates(keep="first", inplace=True)
    df.fillna("Empty", inplace=True)
    df["Total_fail"] = df[failure_list].aggregate("sum", axis=1)

    df, _ = categorical_to_numerical(df)    

    # s_scaler = sklearn.preprocessing.StandardScaler()
    
    # dfx = df.drop(columns=["Car ID", "Total_fail"], axis=1)
    # X_scaled = s_scaler.fit_transform(dfx)
    # dfx_scaled = pd.DataFrame(X_scaled, columns = numerical_feature[:-1])
    return df



class predict_failure:
    def __init__(self, columns_x, column_y, database_path=None, csv_path=None,  random_seed = 123,) -> None:
        self.csv_path = csv_path
        if csv_path is not None:
            self.df = pd.read_csv(self.csv_path)
        else:            
            con = sqlite3.connect(database_path)
            self.df = pd.read_sql_query('SELECT * from failure', con)
        pass

        random.seed(random_seed)
        self.random_seed = random_seed
        self.X_train = None
        self.X_test = None

    def set_feature_col(self, columns_x):
        self.X = self.df[columns_x]
        if self.X_train is not None:
            self.X_train = self.X_train[columns_x]
        if self.X_test is not None:            
            self.X_test = self.X_test[columns_x]
    
    def set_ground_truth_col(self, column_y):
        self.y = self.df[column_y]

    def clean_data(self):
        if self.csv_path is not None:
            pass

    def train_test_split(self, test_size=0.2, shuffle=True, is_print=True):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=test_size, shuffle=shuffle, random_state=self.random_seed)
        print(f"number of rows for training {len(self.X_train)}")
        print(f"number of rows for test {len(self.X_test)}")
        if is_print:
            print(self.X_train.describe().T)
            print(self.y_train.describe().T)

            print(self.X_test.describe().T)
            print(self.y_test.describe().T)

    def visualsize(self):
        pass

    def set_model(self, model="svm", ols_normalize=True):
        if model == "svm":
            self.model = svm.SVC()
        elif model == 'log_reg':
            self.model = linear_model.LogisticRegression(random_state=self.random_seed)    

    def preprocessing(self):
        self.df = preprocessing(self.df)

    def fit(self):
        self.fitted = self.model.fit(self.X_train,self.y_train)

    def evaluate(self):
        score = self.model.score(self.X_test,self.y_test)
        print(f"model score: {score:0.2f}")
        # print(f"classification report {classification_report(self.y_test, self.y_pred)}")
        self.y_pred = self.model.predict(self.X_test)
        cf_matrix = metrics.confusion_matrix(self.y_pred, self.y_test)        
        print(cf_matrix)
        print(f"Accuracy: {metrics.accuracy_score(self.y_test, self.y_pred):0.2f}")
        print(f"Precision: {metrics.precision_score(self.y_test, self.y_pred):0.2f}")
        print(f"Recall: {metrics.recall_score(self.y_test, self.y_pred):0.2f}")        

if __name__ == "__main__":   
    database_path = "./data/failure.db"
        
    predictor = predict_failure(
        database_path=database_path,
        # csv_path="./data/cleaned_failure.db.csv",
        columns_x=numerical_feature[:-1], column_y=[-1])        
    predictor.preprocessing()

    #---------------------------    
    predictor.set_feature_col(numerical_feature[:-1])
    predictor.set_ground_truth_col(numerical_feature[-1])
    predictor.train_test_split(test_size=0.2, is_print=False)
    
    #---------------------------
    print("----------------")    
    print("Use SVM with full numerical features")
    predictor.set_model(model="svm")
    predictor.fit()
    predictor.evaluate()

    #---------------------------    
    print("----------------")
    print("Use SVM with 2 highly correclated features to check how much accuracy drop")
    predictor.set_feature_col(["Usage_num", "Membership_num"])
    predictor.fit()
    predictor.evaluate()
    #"Mod_yy",  "RPM"

    #---------------------------    
    print("----------------")
    print("Use Logistic regression with 2 highly correclated features")
    predictor.set_model(model="log_reg")
    predictor.fit()
    predictor.evaluate()


    aa = 0
    # predictor.clean_data()

    # db_path = "data/failure.db"
    
   
    # train, test = 
    # pass