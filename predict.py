import pandas as pd, gc, concurrent.futures, warnings,  pickle, numpy as np
warnings.filterwarnings("ignore")

def lag_ma_features(df_1cust):
    #print(cust_id)
    #df_1cust = df_train[df_train['Customer ID'] == cust_id]
    customer_id = df_1cust['Customer ID'].unique()
    if df_1cust.shape[0] != df_1cust[['Date','Quantity']].groupby(['Date']).agg({'Quantity':'sum'}).reset_index().shape[0]:
        print(f'Multiple entries for same date for customer id{customer_id}')

    # creating lag features
    lags = [1,2,3,4,5,6]
    lagcols = [f"quantity_lag_{lag}" for lag in lags]
    for lag, lagcol in zip(lags, lagcols):
        df_1cust.loc[:, lagcol] = df_1cust[ "Quantity"].shift(lag).astype('float16').fillna(0)
    
    # creating moving average features
    windows = [3,7, 15, 30]
    ma_cols = [f"quantity_ma_{window}" for window in windows]
    for window,ma_col in zip(windows,ma_cols):
        df_1cust.loc[:, ma_col] = df_1cust["quantity_lag_1"].transform(
            lambda x: x.rolling(window).mean()).astype('float32').fillna(0)
        
    return df_1cust
    
def create_history_df(df_history_1cust, next_visit_series):
    next_visit_series['ActualOrPredicted'] = 'PREDICTED'
    next_visit_series['Quantity'] =  9999
    df_history_1cust = df_history_1cust.append(next_visit_series).reset_index(drop=True)
    return df_history_1cust


df_test = pd.read_csv('df_test_cc.csv', dtype = {'Customer ID':'object'})
df_history = pd.read_csv('df_train_regress_cc.csv', dtype = {'Customer ID':'object'})

gc.collect()

independent_features_model_in = ['dayofmonth_sin', 'dayofmonth_cos', 'month_sin', 'month_cos',
       'week_sin', 'week_cos', 'dayofweek_sin', 'dayofweek_cos', 'quarter_sin',
       'quarter_cos', 'month_start_mid_end_sin', 'month_start_mid_end_cos',
       'price', 'Customer_Cluster', 'quantity_lag_1', 'quantity_lag_2',
       'quantity_lag_3', 'quantity_lag_4', 'quantity_lag_5', 'quantity_lag_6',
       'quantity_ma_3', 'quantity_ma_7', 'quantity_ma_15', 'quantity_ma_30']

classifier_filename = 'classifier.sav'
classifier = pickle.load(open(classifier_filename, 'rb'))
regressor_filename = 'regressor.sav'
regressor = pickle.load(open(regressor_filename, 'rb'))

def predict_model(in_ser):
    classifier_in = in_ser.values.reshape(1, -1)
    order_category = classifier.predict(classifier_in)
    if order_category[0] == 0:
        #return order_category[0],0
        return 0
    else:
        #regressor_in = np.append(classifier_in,order_category).reshape(1,-1)
        regressor_in = classifier_in
        estimated_qty = regressor.predict(regressor_in)
        #return order_category[0],estimated_qty[0]
        return estimated_qty[0]
    
    
def predict_for_a_cust(cust):
    df_1cust_history = df_history[df_history['Customer ID'] == cust].sort_values(by=['Date']).reset_index(drop=True)
    # even though no such customer in current test data, but still this is a provision in the code
    
    df_1cust_test = df_test[df_test['Customer ID'] == cust].sort_values(by=['Date']).reset_index(drop=True)
    df_1cust_history['ActualOrPredicted'] = 'ACTUAL'
    df_1cust_history['Quantity_Actual'] = df_1cust_history['Quantity']
    
    for row_index in range(df_1cust_test.shape[0]):
    #for row_index in range(1):
        next_visit_series = df_1cust_test.iloc[row_index]
        next_visit_series['ActualOrPredicted'] = 'PREDICTED'
        next_visit_series['Quantity_Actual'] =  next_visit_series['Quantity']
        next_visit_series['Quantity'] = 999
        df_1cust_history = df_1cust_history.append(next_visit_series).reset_index(drop=True)
        df_1cust_history_lag = lag_ma_features(df_1cust_history)
        #oc,qe = predict_model(df_1cust_history_lag.iloc[-1][independent_features_model_in])
        qe = predict_model(df_1cust_history_lag.iloc[-1][independent_features_model_in])
        df_1cust_history_lag.loc[df_1cust_history_lag['Date'] == next_visit_series['Date'], 'Quantity'] = qe
        #df_1cust_history_lag.loc[df_1cust_history_lag['Date'] == next_visit_series['Date'], 'order_category'] = oc
        #df_1cust_history_lag.iloc[-1]['Quantity_Estimated'] = qe
    return df_1cust_history_lag
    
    
def run_parallel():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        customers = df_test['Customer ID'].unique()
        #customers = ['500729644','500734413']
        total_customers = len(customers)
        
        return_dfs = executor.map(predict_for_a_cust, customers)
        
        df_test_customers_preds = pd.DataFrame()
        for iteration_no, return_df in enumerate(return_dfs, start=1):
            print(f'No of Customers processed -- {iteration_no} of {total_customers} | percentage of total customers processed -- {"{:.2f}".format((iteration_no*100)/total_customers)} %')
            df_test_customers_preds = df_test_customers_preds.append(return_df)
            
        df_test_customers_preds.rename(columns={'Quantity':'Quantity_Estimated'}, inplace=True)
        df_test_customers_preds.to_csv('FINAL_TEST_PRED.csv',index=False)


if __name__ == '__main__':
    run_parallel()