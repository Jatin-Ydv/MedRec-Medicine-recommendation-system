from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import LabelEncoder
import ast

le=LabelEncoder()

#load database ====================================================================
sym_des=pd.read_csv("datasets/symptoms_df.csv")
precautions=pd.read_csv("datasets/precautions_df.csv")
workout=pd.read_csv("datasets/workout_df.csv")
descriptions=pd.read_csv("datasets/description.csv")
medications=pd.read_csv("datasets/medications.csv")
diets=pd.read_csv("datasets/diets.csv")

main_dataset=pd.read_csv("datasets/Training.csv")
X=main_dataset.drop("prognosis",axis=1)
y=main_dataset["prognosis"]

y=le.fit_transform(y)
disease_list={index: label for index, label in enumerate(le.classes_)}
#load the model

svc=pickle.load(open("Models/svc.pkl","rb"))
#helper functions

def helper(dis):
    
    desc=descriptions[descriptions['Disease']==dis]['Description']
    desc=" ".join([w for w in desc])
    
    pre=precautions[precautions['Disease']==dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre=[col for col in pre.values]
    
    med=medications[medications['Disease']==dis]['Medication']
    medi=ast.literal_eval(med.values[0])
    die=diets[diets['Disease']==dis]['Diet']
    die=ast.literal_eval(die.values[0])
    
    wrkout=workout[workout['disease']==dis]['workout']
    
    return desc,pre,medi,die,wrkout


# model predicted values
symptom_dict = {column: idx for idx, column in enumerate(X.columns)}

def get_predicted_value(patient_symptoms):
    input_vector=np.zeros(len(symptom_dict))

    for item in patient_symptoms:
        if item not in symptom_dict:
            return None
        input_vector[symptom_dict[item]]=1
    
    return disease_list[svc.predict([input_vector])[0]]


app=Flask(__name__)



#creating routes

@app.route('/')

def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST','GET'])

def predict():
    if request.method=='POST':
        
        symptoms=request.form.get('symptoms')
        
        
        user_symptoms=[s.strip() for s in symptoms.split(',')]
        
        # remove extra characters, if any
        user_symptoms=[sym.strip("[]' ") for sym in user_symptoms]
            
        pro_user_symptoms=[inp.lower().replace(' ', '_') for inp in user_symptoms]
        predicted_disease=get_predicted_value(pro_user_symptoms)
        
        if predicted_disease is None:
            error_message="Your symptoms are either too generic or outside the scope of MedRec!"
            des,pre,med,die,wrkout=None,None,None,None,None
            dis_pre=None
        else:
            des,pre,med,die,wrkout=helper(predicted_disease)
            dis_pre=pre[0]
            error_message=None
        return render_template('index.html',predicted_disease=predicted_disease,dis_des=des,dis_pre=dis_pre,
                               dis_med=med,dis_wrkout=wrkout,dis_diet=die,user_symptoms=user_symptoms,
                               error_message=error_message
                               )
    
    
@app.route('/About')

def about():
    return render_template('about.html')

@app.route('/Contact')

def contact():
    return render_template('contact.html')

@app.route('/Blog')

def blog():
    return render_template('blog.html')

@app.route('/Developer')

def developer():
    return render_template('developer.html')

if __name__=="__main__":
    app.run(debug=True)