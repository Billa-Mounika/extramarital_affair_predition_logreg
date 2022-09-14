import json
import pickle
from unittest import result
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd


app = Flask(__name__)
logregmodel=pickle.load(open('logregmodel.pkl','rb'))
scalar=pickle.load(open('scaling_logreg.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    occupation_array = [1,2,3,4,5,6]
    occupation = data['occupation']
    data2 = {} 
    for i in occupation_array:
        if(i == occupation):
            data1 = {"occ_"+str(occupation) : [1.0]}
        else:  
            dict1 = {"occ_"+str(i): [0.0]}
            data2.update(dict1) 

    df1 = pd.DataFrame(data1)     
    df2 = pd.DataFrame(data2) 
    df_final_occ = pd.concat([df1,df2],axis=1)
    df_final_occ.drop(['occ_1'],axis=1,inplace=True)

    husb_occupation = data['occupation_husb']
    data4 = {} 
    for i in occupation_array:
        if(i == husb_occupation):
            data3 = {"occ_husb_"+str(husb_occupation) : [1.0]}
        else:  
            dict3 = {"occ_husb_"+str(i): [0.0]}
            data4.update(dict3) 

    df3 = pd.DataFrame(data3)     
    df4 = pd.DataFrame(data4) 
    df_final_occ_husb = pd.concat([df3,df4],axis=1)
    df_final_occ_husb.drop(['occ_husb_1'],axis=1,inplace=True)
    df_final = pd.concat([df_final_occ,df_final_occ_husb],axis=1)
    df_final['Intercept'] = 1.0

    del data['occupation']
    del data['occupation_husb']

    input_df = pd.DataFrame(data, index=[0])
    df = pd.concat([df_final,input_df],axis=1)
    
    new_data=scalar.transform(df)
    output=logregmodel.predict(new_data)
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    
    occupation_array = [1,2,3,4,5,6]
    #data=[float(x) for x in request.form]
    data=request.form.to_dict(flat=False)

    occupation = request.form['occupation']
    data2 = {} 
    for i in occupation_array:
        if(i == int(occupation)):
            data1 = {"occ_"+str(occupation) : [1.0]}
        else:  
            dict1 = {"occ_"+str(i): [0.0]}
            data2.update(dict1) 
    df1 = pd.DataFrame(data1) 
    df2 = pd.DataFrame(data2)   
     
    df_final_occ = pd.concat([df1,df2],axis=1)
    df_final_occ.drop(['occ_1'],axis=1,inplace=True)

    husb_occupation = request.form['occupation_husb']
    data4 = {} 
    for i in occupation_array:
        if(i == int(husb_occupation)):
            data3 = {"occ_husb_"+str(husb_occupation) : [1.0]}
        else:  
            dict3 = {"occ_husb_"+str(i): [0.0]}
            data4.update(dict3) 

    df3 = pd.DataFrame(data3)     
    df4 = pd.DataFrame(data4) 
    df_final_occ_husb = pd.concat([df3,df4],axis=1)
    df_final_occ_husb.drop(['occ_husb_1'],axis=1,inplace=True)
    df_final = pd.concat([df_final_occ,df_final_occ_husb],axis=1)
    df_final['Intercept'] = 1.0

    del data['occupation']
    del data['occupation_husb']

    input_df = pd.DataFrame(data, index=[0])
    df = pd.concat([df_final,input_df],axis=1)
    
    data=[x for x in df.values.tolist()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    output=logregmodel.predict(final_input)[0]
    if(output == 0.0):
        return render_template("home.html",prediction_text="Dosen't Have Extra Marital Affair")
    else:
        return render_template("home.html",prediction_text="Has Extra Marital Affair")    
    
#if __name__=="__main__":
#    app.run(debug=True) 
if __name__=="__main__":
    app.run(host='0.0.0.0',port=8080)  