import streamlit as st
from ucimlrepo import fetch_ucirepo 
import joblib
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
X = aids_clinical_trials_group_study_175.data.features 
aids_variables = aids_clinical_trials_group_study_175.variables
n_variables = len(aids_variables)

feature_display_name_dict = dict(
    zip(
        ['time', 'trt', 'age', 'wtkg', 'hemo', 'homo', 'drugs',
        'karnof', 'oprior', 'z30', 'zprior', 'preanti', 'race', 'gender',
        'str2', 'strat', 'symptom', 'treat', 'offtrt', 'cd40', 'cd420',
        'cd80', 'cd820'],
        ['Time', 'TRT', 'Age', 'Weight (kgs)', 'Hemophiliac', 'Homosexuality', 'Drugs',
        'Karnof', 'O Prior', 'z30', 'zprior', 'Preanti', 'Race', 'Gender',
        'Str2', 'Strat', 'Symptomatic', 'Treated', 'Offtrt', 'cd40', 'cd420',
        'cd80', 'cd820']
    )
)

st.markdown(
    """
    # Feature Distribution 📊

    First, we will start to visualize the data to gain a better understanding of the UCI AIDS dataset.

    **Select a feature from the list below** and a histogram will be plotted to **depict the overall distribution** of that feature inside the dataset.

    Based on whether these distributions looking Gaussian, Poisson, etc., we will undertake different normalization strategies to deal with them. 
    """
)

feature_select = st.selectbox(
    label="Feature", 
    options=feature_display_name_dict.keys(),
    format_func=lambda x: feature_display_name_dict[x],
    key="feat",
)

if feature_select:  
    fig = plt.figure(figsize=(20,20))
    heights, bin_left_edge, _ = plt.hist(X[feature_select])
    st.pyplot(fig)

