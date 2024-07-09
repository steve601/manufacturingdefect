import os
import sys
import numpy as np
import pandas as pd
from source.exception import UserException
from source.logger import logging
from sklearn.preprocessing import StandardScaler
from source.commons import save_object
from imblearn.over_sampling import RandomOverSampler
from dataclasses import dataclass

# let's create the path of what will be returned i.e preprocessor
@dataclass
class DataTransformationConfig:
    scaler_obj_path:str = os.path.join('elements','scaler.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()
    logging.info('Obtaining scaler object')    
    def get_scaler_obj(self):
        try:
            scaler = StandardScaler()
            
            return scaler
        
        except Exception as e:
            raise UserException(e,sys)
        
    def get_oversampler_obj(self):
        try:
            oversampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
            return oversampler
        except Exception as e:
            raise UserException(e,sys)

    logging.info('Fitting and transforming the data using scaler')    
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            # reading the data
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Successfully read trin and test set')
            logging.info('Obtaining scaler object')
            
            scaler_obj = self.get_scaler_obj()
            oversampler_obj = self.get_oversampler_obj()
            target_column = 'defectstatus'
            
            logging.info('Separating into independent and dependent features for both sets')
            
            input_feature_train_df = train_df.drop(target_column,axis=1)
            output_feature_train_df = train_df[target_column]
            input_feature_train_df,output_feature_train_df = oversampler_obj.fit_resample(input_feature_train_df, output_feature_train_df)
            
            input_feature_test_df = test_df.drop(target_column,axis=1)
            output_feature_test_df = test_df[target_column]
            input_feature_test_df,output_feature_test_df = oversampler_obj.fit_resample(input_feature_test_df, output_feature_test_df)
            
            logging.info('Fitting preprocessor object on the data')
         
            input_feature_train_arr = scaler_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = scaler_obj.transform(input_feature_test_df)
            
            #train_arr = np.c_[input_feature_train_arr,np.array(output_feature_train_df)]
            #test_arr = np.c_[input_feature_test_arr,np.array(output_feature_test_df)]
            
            save_object(
                file_path = self.data_trans_config.scaler_obj_path,
                obj = scaler_obj)
            
            return(
                input_feature_train_arr,
                input_feature_test_arr,
                output_feature_train_df,
                output_feature_test_df
            )
            
        except Exception as e:
            raise UserException(e,sys)
            
            