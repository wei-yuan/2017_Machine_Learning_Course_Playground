import pandas as pd
from pandas import read_csv
from datetime import datetime
from sklearn import preprocessing

########################
#-- data preprocessing-#
########################

# read data
df = pd.read_csv('mini_sample.csv', 
                usecols=['CALL_TYPE', 'TAXI_ID', 'TIMESTAMP', 'POLYLINE'])
# rearrange column here
df_reorder = df[['TIMESTAMP', 'CALL_TYPE', 'TAXI_ID', 'POLYLINE']] 

# TIMESTAMP
def time_prep(dataset):	
    # change timestamp from unix timestamp to hour
    for index, element in enumerate(df_reorder.TIMESTAMP.astype('float32')):                        
        utc_time = datetime.utcfromtimestamp(element)                
        #print ("index: %s, utc_time: %s" % (index, utc_time))
        dataset.loc[index, 'TIMESTAMP'] = int(str(utc_time)[11:13])

def call_prep(dataset):
    call_type = dataset.CALL_TYPE
    le = preprocessing.LabelEncoder()
    le.fit(call_type)
    label_type = le.transform(call_type)    
    max_label = max(label_type).astype('float32')            
    for index, element in enumerate(label_type):                                        
        norm_label = label_type[index].astype('float32') / max_label
        dataset.loc[index, 'CALL_TYPE'] = norm_label
        #print (index, norm_label)        

# TAXI_ID
def id_prep(dataset):
    taxi_id = dataset.TAXI_ID
    # label encoding
    le = preprocessing.LabelEncoder()
    le.fit(taxi_id)
    label_taxi_id = le.transform(taxi_id)    
    max_id = max(label_taxi_id).astype('float32')
    for index, element in enumerate(label_taxi_id):                
        norm_taxi_id = label_taxi_id[index].astype('float32') / max_id
        #print ("norm_taxi_id: %s" % norm_taxi_id)
        dataset.loc[index, 'TAXI_ID'] = norm_taxi_id        

# POLYLINE
def poly_prep(dataset):
    poly = dataset.POLYLINE	
    for index, mini_list in enumerate(poly):
        '''
        print ("# %s OF TAXI TRIP" % (index + 1))
        print ("index: %s, mini_list: %s" % (index, mini_list))
        print "\n"    
        print ("num of point: %s" % len(eval(mini_list)))
        '''
        dataset.loc[index, 'POLYLINE'] = len(eval(mini_list))
        '''
        for index, point in enumerate(eval(mini_list)):                        			
            #print ("index: %s, point: %s" % (index, point))
            #print "\n"
        '''

time_prep(df_reorder)
id_prep(df_reorder)
poly_prep(df_reorder)
call_prep(df_reorder)

print "\n"
# change header name
df_reorder.columns = ['HOUR', 'CALL_TYPE', 'TAXI_ID', 'POLYLINE']
# summarize first 5 rows
print(df_reorder.head(5))
# save to file
df_reorder.to_csv('reordered.csv', index=False)
