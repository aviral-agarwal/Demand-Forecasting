Used pipenv to create a virutal environment so that I can share the exact libraries used to complete the given ask.
to use a different a different virutal environment with jupyter notebook, ipykernel package needs to be installed

# in the virtual environment
pipenv install ipykernel
python -m ipykernel install --user --name=<virtual environment name>

Data_Assignment.csv should be in same directory as code

1. 
run Initial_Data_Preparation.ipynb

to prepare time features, price variable. 
Separate train and test data
logic to create customer clusters

-- will create --
df_train.csv
df_train_regress.csv
df_test.csv
customer_cluster_train.csv
df_train_zero_qty_customers.csv

2.
run Assigning_customer_clusters.ipynb

will assign customer cluster to each data point, in train and test data both. The logic is explained

-- will create --
df_train_regress_cc.csv
df_test_cc.csv

3. 
run generate_lag_ma_features.py from shell

will use all cores of machine and create lag and moving average features in train data
will take time depending on machine hardware, CPU usage will be 100%, machine might heat up too
status is visible in shell

-- will create --
df_train_regress_lag.csv


4. 
run Modelling_Classification and Regression.ipynb
to train classification and regression models

-- will create --
classifier.sav
regressor.sav

5. 
run predict.py from shell

this will also run in parallel to generate output with predicted value
 
-- will create --
FINAL_TEST_PRED.csv