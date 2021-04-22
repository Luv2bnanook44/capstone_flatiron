import pickle

def get_prediction(feature_values):
    """ Given a list of feature values, return a prediction made by the model"""
    
    loaded_model = un_pickle_model()
    
    # Model is expecting a list of lists, and returns a list of predictions
    predictions = loaded_model.predict(feature_values)
    # We are only making a single prediction, so return the 0-th value
    return predictions[0]

def un_pickle_model():
    """ Load the model from the .pkl file """
    with open("src/models/model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model
