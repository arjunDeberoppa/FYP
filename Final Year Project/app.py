# from flask import Flask, render_template, request
# import pickle
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load the trained model
# with open('model.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

# @app.route('/')
# def home():
#     return render_template('prediction.html')

# @app.route('/submit', methods=['POST'])
# def submit():
#     # Load the one-hot encoding transformation
#     with open('one_hot_encoder.pkl', 'rb') as encoder_file:
#         one_hot_encoder = pickle.load(encoder_file)

#     # Extracting user inputs from the form
#     features = [int(request.form[f'q{i}']) for i in range(1, 11)]
#     gender = request.form['gender']
#     ethnicity = request.form['ethnicity']
#     jundice = request.form['jundice']
#     austim = request.form['austim']
#     country_of_res = request.form['country_of_res']
#     used_app_before = request.form['used_app_before']
#     relation = request.form['relation']

#     # Encoding categorical variables using the loaded one-hot encoder
#     gender_encoded = 1 if gender == 'm' else 0
#     ethnicity_encoded = one_hot_encoder['ethnicity'].transform(pd.DataFrame({'ethnicity': [ethnicity]})).values[0]
#     jundice_encoded = 1 if jundice == 'yes' else 0
#     austim_encoded = 1 if austim == 'yes' else 0
#     country_of_res_encoded = one_hot_encoder['country_of_res'].transform(pd.DataFrame({'country_of_res': [country_of_res]})).values[0]
#     used_app_before_encoded = 1 if used_app_before == 'yes' else 0
#     relation_encoded = one_hot_encoder['relation'].transform(pd.DataFrame({'relation': [relation]})).values[0]

#     # Combine all features
#     input_data = features + [gender_encoded, *ethnicity_encoded, jundice_encoded, austim_encoded,
#                             *country_of_res_encoded, used_app_before_encoded, *relation_encoded]

#     # Make prediction
#     prediction = model.predict(np.array([input_data]))
#     result = "YES" if prediction[0][0] > 0.5 else "NO"


#     return render_template('result.html', prediction=result)

# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd  # Make sure to import pandas

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the one-hot encoding transformation
with open('one_hot_encoder.pkl', 'rb') as encoder_file:
    one_hot_encoder = pickle.load(encoder_file)

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Extracting user inputs from the form
    features = [int(request.form[f'q{i}']) for i in range(1, 11)]
    gender = request.form['gender']
    ethnicity = request.form['ethnicity']
    jundice = request.form['jundice']
    austim = request.form['austim']
    country_of_res = request.form['country_of_res']
    used_app_before = request.form['used_app_before']
    relation = request.form['relation']

    # Encoding categorical variables using the loaded one-hot encoder
    gender_encoded = 1 if gender == 'm' else 0
    ethnicity_encoded = one_hot_encoder.transform(pd.DataFrame({'ethnicity': [ethnicity]})).values[0]
    jundice_encoded = 1 if jundice == 'yes' else 0
    austim_encoded = 1 if austim == 'yes' else 0
    country_of_res_encoded = one_hot_encoder.transform(pd.DataFrame({'country_of_res': [country_of_res]})).values[0]
    used_app_before_encoded = 1 if used_app_before == 'yes' else 0
    relation_encoded = one_hot_encoder.transform(pd.DataFrame({'relation': [relation]})).values[0]

    # Combine all features
    input_data = features + [gender_encoded, *ethnicity_encoded, jundice_encoded, austim_encoded,
                            *country_of_res_encoded, used_app_before_encoded, *relation_encoded]

    # Apply the same one-hot encoding transformation to user input
    input_data = one_hot_encoder.transform(pd.DataFrame([input_data]))

    # Make prediction
    prediction = model.predict(np.array(input_data))
    result = "YES" if prediction[0][0] > 0.5 else "NO"

    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)

