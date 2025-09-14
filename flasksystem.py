from flask import Flask, request, render_template


import os
from dashboardandneuralnetwork.resume_parser import extract_text_from_pdf, extract_skills
from dashboardandneuralnetwork.job_fetcher import fetch_jsearch_jobs 

from predict_role import predict_role_from_text


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':

        try:
            # save and uploadthe resumee
            file = request.files['resume']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            
            file.save(file_path)

            #extractionof  text from resume
            text = extract_text_from_pdf(file_path)
            #Predict role based on model
            predicted_role = predict_role_from_text(text)

            #extract skills
            skills = extract_skills(text)

            #get jobs from jSearchapi
             
            jobs = fetch_jsearch_jobs(skills[0]) if skills else fetch_jsearch_jobs(predicted_role)


            return render_template(
                'results.html',
                skills=skills,
                jobs=jobs,


                predicted_role=predicted_role.title(),
                error=None


            )
        except Exception as e:

            return render_template(
                'results.html',
                skills=[],
                jobs=[],


                predicted_role="",

                error=str(e)
            )
    return render_template('index.html')
#Dashboard location

@app.route('/dashboard')
def dashboard():


    return render_template('dashboard.html')
if __name__ == '__main__':

    app.run(debug=True)
