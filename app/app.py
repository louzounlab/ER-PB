import datetime
import secrets
import os
from os.path import join
from flask import Flask, render_template, request
import time
import pickle
import shutil
import pandas as pd


# create an instance of the Flask class, with the name of the running application and the paths for the static files and templates
app = Flask(__name__, static_folder='static', template_folder="templates")

# set the upload folder to the absolute path of the "upload_folder" directory
app.config['UPLOAD_FOLDER'] = os.path.abspath("upload_folder")

# set the lifetime of a session to one hour
app.config["PERMANENT_SESSION_LIFETIME"] = datetime.timedelta(hours=1)

# set the secret key to a random string generated by the secrets module
app.config["SECRET_KEY"] = secrets.token_hex()

# Load the models once
models_names = ["xgboost_2.pkl", "xgboost_7.pkl", "xgboost_34.pkl"]
loaded_models = []
for model_name in models_names:
    with open(join("models", model_name), "rb") as file:
        model = pickle.load(file)
        loaded_models.append(model)


def clean_old_files():
    files = os.listdir("static")
    for file in files:
        if os.path.isdir(join("static", file)) and file != "bootstrap":
            # Time is older than an hour
            if time.time() - float(file) > 3600:
                shutil.rmtree(join("static", file))
                print("Deleted:", file)


@app.route('/process_form', methods=['POST', 'GET'])
def process_form():
    try:
        # Get the form data
        data = request.form

        # Get the median data dataframe
        data_df = pd.read_csv(join("static", "initial_df.csv"))

        # Fill the data from the form into the median data format
        if data["gest_age_weeks"] and data["gest_age_days"]:
            data_df["Gestational age at admission"] = float(data["gest_age_weeks"])+float(data["gest_age_days"])/7
        if data['parity']:
            data_df['Parity'] = float(data['parity'])
        data_df["Gestational hypertensive disorders"] = int(data["ges_hype_dis"])
        if data["max_pulse"]:
            data_df["Maximal pulse at admission"] = float(data["max_pulse"])
        data_df['Previous hospitalizations during pregnancy'] = int(data['prev_hos'])
        data_df['Premature preterm rupture of membranes'] = int(data['pprom'])
        data_df['Cervical dynamics'] = int(data['cervical_dynamics'])
        if data['amniotic_fluid_index']:
            data_df['Amniotic Fluid Index at admission'] = float(data['amniotic_fluid_index'])
        if data['cervical_dilation']:
            data_df['Cervical dilation'] = float(data['cervical_dilation'])
        if data['hemoglobin']:
            data_df['Hemoglobin at admission'] = float(data['hemoglobin'])

        # Predict the risk
        risks = []
        for model in loaded_models:
            risks.append(model.predict_proba(data_df)[:, 1][0])
        risks = [str(round(float(risk*100), 2))+'%' for risk in risks]

        return render_template("index.html", active="Home", risks=risks)
    except Exception as e:
        return render_template("index.html", active="Home", risks=[], error=str(e))


@app.route('/', methods=['GET'])
@app.route('/Home', methods=['GET'])
def home():
    return render_template("index.html", active="Home", risks=None)


@app.route('/Example', methods=['GET'])
def example():
    return render_template("example.html", active="Example")


@app.route('/About', methods=['GET'])
def about():
    return render_template("about.html", active="About")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=True)
