from flask import Flask,request,render_template
from source.logger import logging
from source.main_project.pipeline.predict_pipeline import PredicPipeline,UserData

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('defect.html')

@app.route('/predict',methods = ['POST'])
def do_prediction():
    user_info = UserData(
        productionvolume=request.form.get('vol'),
        productioncost=request.form.get('cost'),
        supplierquality=request.form.get('qual'),
        defectrate=request.form.get('rate'),
        qualityscore=request.form.get('score'),
        maintenancehours=request.form.get('hrs'),
        stockoutrate=request.form.get('stock'),
        safetyincidents=request.form.get('safety'),
        energefficiency=request.form.get('energy')
    )
    logging.info('Converting values to a pandas df')
    user_df = user_info.get_data_as_df()
    logging.info('Initializing predict pipeline class')
    predictpipe = PredicPipeline()
    
    y_pred = predictpipe.predict(user_df)
    
    msg = 'Defects are likely to be low' if y_pred == 0 else 'Defects are likely to be high'
    
    return render_template('defect.html',text=msg)

if __name__ == "__main__":
    app.run(host="0.0.0.0")