import pickle

def un_pickle_model(model_path):
    """ Load the model from the .pkl file """
    with open(model_path, "rb") as model_file:
        loaded_model = pickle.load(model_file)
    return loaded_model

def get_prediction(feature_values, model_path):
    """
    Given a list of feature values,
    return a prediction made by the model
    """

    loaded_model = un_pickle_model(model_path)

    # Model is expecting a list of lists, and returns a list of predictions
    predictions = loaded_model.predict(feature_values)
    probs = loaded_model.predict_proba(feature_values)
    # We are only making a single prediction, so return the 0-th value
    return [predictions[0], probs[0]]

