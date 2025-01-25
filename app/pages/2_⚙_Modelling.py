import os.path
import pickle

import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.functional.feature import detect_feature_types, FeatureType
from autoop.core.ml.artifact import Artifact

from autoop.core.ml import model, metric
from autoop.core.ml.pipeline import Pipeline

# Set the page configuration
st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str) -> None:
    """
    Helper function to display helper text with custom styling.
    Args:
        text (str): Text to be displayed.
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine "
                  "learning pipeline to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")
if not datasets:
    st.warning("No datasets found. Please upload a dataset to proceed.")
    st.stop()
else:
    st.write("## Available Datasets")
    write_helper_text("Select a dataset from the dropdown below to load it.")

    # Create a select box for dataset selection
    dataset_names = [dataset.name for dataset in datasets]
    selected_dataset_name = st.selectbox("Select a dataset", dataset_names)

    # Find the selected dataset object
    selected_dataset_art = next((ds for ds in datasets
                                 if ds.name == selected_dataset_name), None)

    selected_dataset = Dataset(asset_path=selected_dataset_art.asset_path,
                               name=selected_dataset_art.name,
                               version=selected_dataset_art.version,
                               data=selected_dataset_art.data,
                               tags=selected_dataset_art.tags,
                               metadata=selected_dataset_art.metadata)

    features = detect_feature_types(selected_dataset)

    feature_names = [feature.name for feature in features]
    input_feature_names = st.multiselect("Select Input Features",
                                         feature_names)
    target_feature_name = st.selectbox("Select Target Feature", feature_names)

    input_features = [feature for feature in features
                      if feature.name in input_feature_names]
    target_feature = [feature for feature in features
                      if feature.name == target_feature_name][0]

    if target_feature_name in input_feature_names:
        st.warning("OVER-I would not include the target as input, "
                   "but it's your party!-FITTING")

    if target_feature.type == FeatureType.CATEGORICAL:
        task = "CLASSIFICATION"
    else:
        task = "REGRESSION"
    write_helper_text(f"The selected features are for {task}.")

    write_helper_text(f"These are the available models for {task}:")

    model_names = model.REGRESSION_MODELS if task == "REGRESSION" \
        else model.CLASSIFICATION_MODELS
    selected_model_name = st.selectbox("Select a model", model_names)

    selected_model = model.get_model(selected_model_name)

    # Add a slider for train-test split
    split_ratio = st.slider(
        "Select Train-Test Split Ratio",
        min_value=0.,
        max_value=1.,
        value=0.8,
        step=0.1,
    )

    split_ratio_int = int(100 * split_ratio)

    # Display the selected split ratio
    st.write(f"Training data: {split_ratio_int}%")
    st.write(f"Testing data: {100 - split_ratio_int}%")

    # Display metrics
    write_helper_text("These are the available metrics for the given tasks:")
    metrics_names = metric.METRICS_REGRESSION if task == "REGRESSION" \
        else metric.METRICS_CLASSIFICATION
    selected_metric_names = st.multiselect("Select evaluating metrics",
                                           metrics_names)

    selected_metrics = [metric.get_metric(_metric) for _metric
                        in selected_metric_names]

    st.write("### Pipeline Summary")
    st.markdown(f"""
    **Dataset:** {selected_dataset_name}
    **Input Features:** {', '.join(input_feature_names)}
    **Target Feature:** {target_feature_name}
    **Task Type:** {task}
    **Model Selected:** {selected_model_name}
    **Train-Test Split:** {split_ratio_int}% Training /
            {100 - split_ratio_int}% Testing
    **Metrics Selected:** {', '.join(selected_metric_names)}
    """)

    execute_button = st.button("Execute")

    if execute_button:
    exec_condition = (
        (not selected_metrics == [])
        and (selected_dataset is not None)
        and (selected_model is not None)
        and (not input_features == [])
        and (target_feature is not None)
    )

        if exec_condition:
            pipeline = Pipeline(selected_metrics,
                                selected_dataset,
                                selected_model,
                                input_features,
                                target_feature,
                                split_ratio)

            results = pipeline.execute()

            # Store the pipeline in Streamlit's session state dictionary
            # So it remains available at each rerun of the script
            st.session_state["pipeline"] = pipeline

            training_metrics = results["metrics on training set"]
            testing_metrics = results["metrics on evaluation set"]
            preds = results["predictions"]

            st.write("## Results")

            st.write("### Metrics on Training Set:")
            for metric_fn, value in training_metrics:
                st.write(f"**{metric_fn.__class__.__name__}:** {value:.4f}")

            st.write("### Metrics on Evaluation Set:")
            for metric_fn, value in testing_metrics:
                st.write(f"**{metric_fn.__class__.__name__}:** {value:.4f}")

            st.session_state["pipeline"] = pipeline

        else:
            write_helper_text("Execution condition violated. "
                              "Please configure the pipeline first.")

    # Check if pipeline exists, and save it
    if "pipeline" in st.session_state:
        pipeline = st.session_state["pipeline"]

        pipeline_name = st.text_input("Pipeline name", "my_pypeline")
        pipeline_version = st.text_input("Pipeline version", "0.0")
        pipeline_path = st.text_input(
            "Path where to save",
            os.path.join(
                os.path.curdir,
                pipeline_name + pipeline_version
            )
        )

        if st.button("Save Pipeline"):
            art = Artifact(asset_path=pipeline_path, name=pipeline_name,
                           version=pipeline_version,
                           data=pickle.dumps(pipeline),
                           tags=[], metadata={}, type_="pipeline")

            AutoMLSystem.get_instance().registry.register(art)
            st.success("Pipeline saved successfully.\n" + str(pipeline))
