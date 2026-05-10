import shap
import numpy as np

def explain_model(model, X, feature_names):
explainer = shap.Explainer(model, X)
shap_values = explainer(X)
return shap_values
