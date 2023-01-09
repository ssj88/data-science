import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

#%matplotlib inline

sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')

diabetes_data = pd.read_csv("G:/2.MCA Course/20MCA241 DATA SCIENCE LAB/LAB/EXERCISE_5/diabetes.csv")
# View top 5 rows of our dataset
diabetes_data.head()