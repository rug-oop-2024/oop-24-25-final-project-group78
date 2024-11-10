import streamlit as st
# import pandas as pd

from app.core.system import AutoMLSystem
# from autoop.core.ml.dataset import Dataset
# from autoop.functional.feature import detect_feature_types


# Set the page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """Helper function to display helper text with custom styling.
    Args:
        text (str): Text to be displayed.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine "
                  "learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
