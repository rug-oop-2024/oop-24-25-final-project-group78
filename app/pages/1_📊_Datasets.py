import sys
import os
import streamlit as st
import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# sys.path.insert(0,
                # os.path.abspath
                # ("C:/Users/Bianca/Desktop/OOP 2/"
                 # "oop-24-25-final-project-group_80"))
"""
I used this because my system was not
using the right path to the app.
"""


st.set_page_config(page_title="Manage Datasets", page_icon="📊")


def app() -> None:
    """Streamlit application function to manage datasets."""

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")
    print(datasets)

    st.title('Manage the datasets')

    uploaded_file = st.file_uploader('Choose a CSV file', type='csv')

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of the uploaded dataset: " + uploaded_file.name)
        st.write(df.head())

        dataset = Dataset.from_dataframe(df,
                                         name=uploaded_file.name,
                                         asset_path=uploaded_file.name)

        AutoMLSystem.get_instance().registry.register(dataset)


app()
