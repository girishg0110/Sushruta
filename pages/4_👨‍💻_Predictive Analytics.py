import streamlit as st
from ucimlrepo import fetch_ucirepo 
import joblib
import numpy as np

clf = joblib.load("clf.joblib")

aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
X = aids_clinical_trials_group_study_175.data.features 
aids_variables = aids_clinical_trials_group_study_175.variables
n_variables = len(aids_variables)

st.markdown(
    """
    # Predictive Analytics 👨‍💻
    As part of Sushruta's diagnostic offerings, a **highly interpretable linear regression model** has been trained on UCI's AIDS dataset. 

    By finetuning the parameters below, you can generate predictions of how likely a certain profile is to have contracted AIDS, based on a variety of features.

    Overall, this model has an accuracy of 83.18% on the UCI dataset. The scoring functions and other related training code are available in the iPython notebook training.ipynb
    """
)

def make_prediction():
    X = []
    for var in aids_variables["name"][2:]:
        X.append(int(st.session_state[var]))
    X = np.array([X])
    prediction_prob = clf.score(X, [1])
    with st.expander("Sushruta's prediction"):
        if prediction_prob < 1e-6:
            st.write(f"The likelihood of AIDS infection according to the model is less than 1e-6.")
        else:
            st.write(f"The likelihood of AIDS infection according to the model is {prediction_prob * 100:0.2f}%")

with st.form(key="var_values"):
    for var_idx in range(2, n_variables):
        name, role, demographic = aids_variables["name"][var_idx], \
            aids_variables["type"][var_idx], aids_variables["demographic"][var_idx] 
        label = demographic if demographic else name
        if role == "Integer":
            st.slider(label, min_value = min(X[name]) - 5,
                        max_value = min(X[name]) + 5, key=name)
        elif role == "Binary": 
            st.selectbox(label, options=(1, 0), format_func=lambda x: "Yes" if x == 1 else "No", key=name)
        elif role == "Continuous":
            st.number_input(label, min_value = min(X[name]),
                        max_value = min(X[name]), key=name)
    st.form_submit_button(on_click=make_prediction)
