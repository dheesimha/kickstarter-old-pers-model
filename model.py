import pickle
# import re
# # import pandas as pd
# # import sklearn
# from pathlib import Path
# from datetime import datetime, timezone
import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer

# BASE_DIR = Path(__file__).resolve(strict=True).parent

# with open(f"{BASE_DIR}/model.pkl", "rb") as f:
#     model = pickle.load(f)
# with open(f"{BASE_DIR}/sc.pkl", "rb") as f:
#     sc = pickle.load(f)
# with open(f"{BASE_DIR}/pca.pkl", "rb") as f:
#     pca = pickle.load(f)
# with open(f"{BASE_DIR}/le.pkl", "rb") as f:
#     le = pickle.load(f)
    
model = pickle.load(open('model.pkl','rb'))
sc = pickle.load(open('sc.pkl','rb'))
pca = pickle.load(open('pca.pkl','rb'))
le = pickle.load(open('le.pkl','rb'))
# def calculate_days(date_string):
#     today = datetime.today().date()
#     specified_date = datetime.strptime(date_string, '%Y-%m-%d').date()
#     days_difference = (today - specified_date).days
#     return days_difference

# ALSO CHANGE THE NAME OF THE FILE HERE SHREYAS
# with open(f"{BASE_DIR}/trained_pipeline-1.0.0.pkl","rb") as f:
#     model = pickle.load(f)
    
# result = ["A good company", "Will likely not survive"]

def predict_pipeline(advert,age_fund,age_mile,relation_score,signi_event,second_round,num_employ,top500):
    # processed_category = re.sub(r'\W+', ' ', category).lower()
    # processed_first_funding_date = calculate_days(first_funding_date)
    # processed_last_funding_date = calculate_days(last_funding_date)

    #MAKE THE PANDAS DATAFRAME BELOW THIS USING THE PARAMETERS FROM ABOVE AND ALSO SEE SPECIFICALLY SO THAT EVEN ORDER MATCHES WITH X_TRAIN
    #Mantej, parameters are matching with the X train - Same order

    # processed_last_funding_duration = processed_last_funding_date - processed_first_funding_date
    # Create a feature vector based on the input parameters THE X_test_con IS THE FINAL FROM THE COPY PASTING THE CODE
    # data = {
    # 'category': [category],
    # 'total_funding': [total_funding],
    # 'country_code': [country_code],
    # 'total_funding_rounds': [total_funding_rounds],
    # 'total_funding_rounds1': [total_funding_rounds],
    # 'funding duration' : [processed_last_funding_duration],
    # 'first_funding_date': [processed_first_funding_date],
    # 'last_funding_date': [processed_last_funding_date]
    # }

    # X_train = pd.DataFrame(data)
    if(age_fund == 0 and age_mile == 0 and top500 == 'Yes'):
        suc = 1
        return suc
    
    if(relation_score == 100):
        suc = 1
        return suc
    
    if(signi_event > 9):
        suc = 1
        return suc
        
    int_features = [0]*8
    int_features[1] = age_fund
    int_features[2] = age_mile
    int_features[3] = relation_score
    int_features[4] = signi_event
    int_features[6] = num_employ
    if advert == 'Yes':
        int_features[0] = 1
    else:
        int_features[0] = 0
    if second_round == 'Yes':
        int_features[5] = 1
    else:
        int_features[5] = 0
    if top500 == 'Yes':
        int_features[7] = 1
    else:
        int_features[7] = 0
    final_features = [float(x) for x in int_features]
    final_features = [np.array(int_features)]
    final_features = sc.transform(final_features)
    final_features = pca.transform(final_features)
    prediction = model.predict(final_features)
 
    # prediction = model.predict(X_train_con)
    
    return prediction
