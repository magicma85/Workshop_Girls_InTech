

import streamlit as st
import mlflow
from workshop.pipeline import Pipeline
st.title("Your first ML Data app using streamlit :) ")
"""
## Task 10 (and last!)

Call your favourite model using mlflow from withing streamlit and visualise it in the streamlit web app.
Tip: to run and visualize streamlit run in the terminal:
```
streamlit run workshop/streamlit_ui.py 
```
            
## Solution
        """
run_id = '36e2dd37044147029d0ca24db0494cb2'### YOUR CODE HERE ###
pipeline = Pipeline()
pipeline.model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
#model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
st.markdown(pipeline.model)

content= st.text_input("Type your query", value='I lost my card')
if st.button("Predict"):
    prediction = pipeline.predict(content)###
    st.text(prediction)
