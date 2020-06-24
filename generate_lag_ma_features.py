import pandas as pd, warnings, concurrent.futures

warnings.filterwarnings("ignore")

df_train = pd.read_csv('df_train_regress_cc.csv', dtype={'Customer ID':'object'})
df_train['Date'] = pd.to_datetime(df_train['Date'])

def lag_ma_features(cust_id):
    #print(cust_id)
    df_1cust = df_train[df_train['Customer ID'] == cust_id].sort_values(by=['Date'])
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

def run_parallel():
    with concurrent.futures.ProcessPoolExecutor() as executor:
        customers = df_train['Customer ID'].unique()
        #customers = ['500015112','500015128']
        total_customers = len(customers)
        
        return_dfs = executor.map(lag_ma_features, customers)
        
        df_train_new = pd.DataFrame()
        for iteration_no, return_df in enumerate(return_dfs, start=1):
            print(f'No of Customers processed -- {iteration_no} of {total_customers} | percentage of total customers processed -- {"{:.2f}".format((iteration_no*100)/total_customers)} %')
            df_train_new = df_train_new.append(return_df)
        
        df_train_new.to_csv('df_train_regress_lag.csv',index=False)


if __name__ == '__main__':
    run_parallel()