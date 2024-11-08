import sys
import os
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


# Add the absolute path to the project root directory
sys.path.insert(0,
                os.path.abspath
                ("C:/Users/Bianca/Desktop/OOP 2/"
                 "oop-24-25-final-project-group_80"))


def app():
    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    st.title('Manage the datasets')

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Display the DataFrame for confirmation
        st.write("Preview of the uploaded dataset: " + uploaded_file.name)
        st.write(df.head())

        dataset = Dataset.from_dataframe(df,
                                         name=uploaded_file.name,
                                         asset_path=uploaded_file.name)

        AutoMLSystem.get_instance().registry.register(dataset)


app()
