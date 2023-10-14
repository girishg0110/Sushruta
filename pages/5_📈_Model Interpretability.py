import streamlit as st
import numpy as np
from ucimlrepo import fetch_ucirepo 
import joblib
from matplotlib import pyplot as plt
from itertools import groupby

aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890) 
features = aids_clinical_trials_group_study_175.variables["name"].to_numpy()
n_features = len(features)

clf = joblib.load("clf.joblib")
clf_weights = clf.coef_[0]
n_weights = len(clf_weights)

st.markdown(
    """
    # Model Interpretability 📈
    Finally, let's try to build a deeper understanding of the model. We can get a rough estimate of how important each feature is to the overall prediction in the following manner.

    The linear regression model is made up of a series of "weights". There is exactly one weight for each variable. 

    The magnitude of the weight (whether positive or negative) is **directly correlated with how important it was in the final diagnosis**.

    The weights of the linear regression model are plotted in the bar chart below.
    """
)

fig = plt.figure(figsize=(20,10))
plt.bar(range(n_weights), clf_weights, color = ['green' if w > 0 else 'red' for w in clf_weights], tick_label=features[2:])
fig.axes[0].xaxis.tick_top()

st.pyplot(fig=fig)

grouped_features = {}
for k, g in groupby(range(n_weights), key=lambda f_idx: clf_weights[f_idx] > 0):
    if k not in grouped_features:
        grouped_features[k] = []
    grouped_features[k].extend([features[idx] for idx in g])

st.markdown(
    """
    We can also glean further insight from the classification of weights into positive or negative categories. 

    If a variable is associated with a positive weight, that means its presence makes an AIDS diagnosis **more likely**.

    The following table lists out the features with a positive weight, followed by those with a negative weight. 
    """
)

st.dataframe(grouped_features)

st.markdown(
    """
    Expectedly, advanced age and the presence of hemophilia in the patient's cardiovascular system increases the likelihood of an AIDS diagnosis. 
    
    Surprisingly, however, our model shows a **negative correlation** between high weight and AIDS likelihood. Given more time, this is definitely something I would want to investigate further.        
    """
)