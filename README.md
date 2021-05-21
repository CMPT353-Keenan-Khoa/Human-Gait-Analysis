# gait analysis

CMPT353 Project: gait analysis

# cleaning.py

Just for testing various cleaning functions.

**command: python cleaning.py "name of file"**

**output:**

ave left-right step duration

number of steps

total time taken

pace(step/time)

pace(step/distance)

step length

# balance_test.py

statistical tests to know the mean of right step duration and left step durataion are same

**command: python balance_test.py "name of file"**

**output:**

normality test for right and left step dataset

variance test for right and left step dataset

Ttest result

Utest result

# create_MLdata.py

produces cleaned data in one dataframe and output a csv file

**command: python create_MLdata.py**

**You have to edit inside of the code to add new data csv file, if you want**

**output:**

# machine_learning.py

uses random forest classifier to classfy given data

**command: python machine_learning.py mldata.csv**

(mldata.csv from create_MLdata.py)

**output:**

valid score

prediction with a random data