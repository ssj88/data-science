    #Importing the libraries  
  
import pandas as pd  
      
    # Importing the dataset  
dataset = pd.read_csv('G:/2.MCA Course/20MCA241 DATA SCIENCE LAB/LAB/EXERCISE_6/User_Data.csv')  
x = dataset.iloc[:, [2, 3]].values  
y = dataset.iloc[:, 4].values  
      
    # Splitting the dataset into the Training set and Test set  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)  
      
    # Feature Scaling  
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)  
    
    # Fitting Naive Bayes to the Training set  
from sklearn.naive_bayes import GaussianNB  

classifier = GaussianNB()  
classifier.fit(x_train, y_train)  
    
    # Predicting the Test set results  
y_pred = classifier.predict(x_test)  
    
     # Making the Confusion Matrix  
from sklearn.metrics import confusion_matrix  
cmtx = confusion_matrix(y_test, y_pred)  
print ("CM", cmtx)

# Print the score: the mean accuracy of the method used
print("Gaussian Naive bayes score      :",classifier.score(x_test,y_test))
