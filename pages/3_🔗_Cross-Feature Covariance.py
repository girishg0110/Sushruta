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

st.markdown(
    """
    # Cross-Feature Covariance 🔗
    To gain more intuition into the patterns present in the dataset, cross-feature covariance exploits the correlations in categorical variables in the UCI dataset.

    **Choose any two categorical variables** and view how their values are distributed in the confusion matrix below. 

    Darker colors indicate that a certain combination of variable values is more prominent in the dataset, while lighter hues denote the opposite.
    """
)

categorical_features_name_dict = dict(
    zip(
        ["trt", "hemo", "homo", "drugs", "oprior", "z30", "gender", "str2", "strat", "symptom", "treat", "offtrt"],
        ["TRT", "Hemophilic", "Homosexuality", "Drug Use", "O Prior", "z30", "Gender", "Str2", "Strat", "Symptomatic", "Treated", "Off TRT"]
    )
)

feature_select = st.sidebar.multiselect(
    label="features", 
    options=categorical_features_name_dict.keys(),
    format_func=lambda x: categorical_features_name_dict[x],
    key="features",
    max_selections=2,
    default=["drugs", "symptom"]
)

if feature_select and len(feature_select) == 2:  
    data_feature_0 = X[feature_select[0]]
    data_feature_1 = X[feature_select[1]]
    cm = confusion_matrix(data_feature_0, data_feature_1)
    fig = plt.figure(figsize=(20,20))    
    sns.heatmap(
        cm, 
        annot=True,
        fmt='g', 
        xticklabels=list(set(data_feature_0)),
        yticklabels=list(set(data_feature_1))
    ) 
    st.pyplot(fig)