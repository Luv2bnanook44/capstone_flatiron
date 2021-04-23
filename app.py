from flask import Flask, send_from_directory, render_template, request, redirect, url_for 
from waitress import serve
from src.utils import extract_feature_values 
from src.models.predictor import get_prediction

app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():    
    """Return the main page."""    
    return send_from_directory("static", "index.html")

@app.route("/make_prediction", methods=["POST"])
def make_prediction():    
    """ Use the ML model to make a prediction using the form inputs. """
   # Get the data from the submitted form    
    data = request.form    
    print(data) # Remove this when you're done debugging
    # Convert the data into just a list of values to be sent to the model    
    feature_values = extract_feature_values(data)    
    # Send the values to the model to get a prediction    
    prediction, probs = get_prediction(feature_values)

    if prediction == 0:
        prediction = "HAVE NOT"
    elif prediction == 1:
        prediction = "HAVE"

    prob0 = round(probs[0]*100, 3)
    prob1 = round(probs[1]*100, 3)


    # Tell the browser to fetch the results page, passing along the prediction    
    return redirect(url_for("show_results", prediction=prediction, prob0=prob0, prob1=prob1))

@app.route("/show_results")
def show_results():    
    """ Display the results page with the provided prediction """        
    # Extract the prediction from the URL params    
    prediction = request.args.get("prediction")
    prob0 = request.args.get("prob0")
    prob1 = request.args.get("prob1")
       
    # prediction = round(float(prediction), 3)
    # Return the results pge    
    return render_template("results.html", prediction=prediction, prob0=prob0, prob1=prob1)
    
if __name__ == "__main__":    serve(app, host='0.0.0.0', port=5000)