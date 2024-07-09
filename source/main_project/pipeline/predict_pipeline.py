import sys
import pandas as pd
from source.commons import load_object
from source.exception import UserException
from source.logger import logging

class PredicPipeline:
    def __init__(self):
        pass
    
    logging.info('Preprocessing user input and making predictions')
    def predict(self,features):
        model_path = 'elements\model.pkl'
        scaler_path = 'elements\scaler.pkl'
        # loaeding objects
        model = load_object(model_path)
        scaler = load_object(scaler_path)
        
        data_scaled = scaler.transform(features)
        prediction = model.predict(data_scaled)
        
        return prediction
logging.info('This class is responsible for mapping all the inputs from html to flask')
class UserData:
    def __init__(self,productionvolume, productioncost,supplierquality,defectrate,
       qualityscore,maintenancehours,stockoutrate,safetyincidents,energefficiency):
        
        self.vol = productionvolume
        self.cost = productioncost
        self.qual = supplierquality
        self.rate = defectrate
        self.score = qualityscore
        self.hrs = maintenancehours
        self.stock = stockoutrate
        self.safety = safetyincidents
        self.energy = energefficiency
    logging.info("Converting user's input to df")  
    # let's write a function that returns the user input as a pandas dataframe
    def get_data_as_df(self):
        try:
            columns = ['productionvolume', 'productioncost', 'supplierquality', 'defectrate',
                    'qualityscore', 'maintenancehours', 'stockoutrate', 'safetyincidents',
                    'energyefficiency']
            user_data = [[self.vol,self.cost,self.qual,self.rate,self.score,self.hrs,self.stock,self.safety,self.energy]]
            return pd.DataFrame(user_data,columns=columns)
        except Exception as e:
            raise UserException(e,sys)
        