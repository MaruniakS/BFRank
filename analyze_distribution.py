import shap
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from constants import FEATURES, SEQUENCE_LENGTH

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

def feature_contribution_distribution(shap_values, feature_names):
    """
    Visualize the distribution of SHAP values for each feature as a violin plot.

    Parameters:
        shap_values: SHAP values, shape (n_samples, n_features).
        feature_names: List of feature names.
    """
    # Convert SHAP values to a DataFrame for visualization
    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    # Plot the distribution using a violin plot
    plt.figure(figsize=(16, 8))
    sns.violinplot(data=shap_df, scale="width", inner="quartile")
    plt.title("Feature Contribution Distributions")
    plt.xlabel("Features")
    plt.ylabel("SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def explain_with_shap_violin(model, x_data, feature_names, sequence_length):
    """
    Compute SHAP values and visualize the distribution of feature contributions.

    Parameters:
        model: Trained TensorFlow model.
        x_data: Input data (n_samples, sequence_length, n_features) as a list or array.
        feature_names: List of feature names.
        sequence_length: Number of time steps.
    """
    n_features = len(feature_names)
    x_data = np.array(x_data)
    x_data_flattened = x_data.reshape(x_data.shape[0], -1)

    # Use the first 10 samples as background
    background_data = x_data_flattened[:10]
    test_data = x_data_flattened[:100]  # Increase this to analyze a larger subset

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, sequence_length, n_features)),
        background_data
    )

    # Compute SHAP values for the test data
    shap_values = explainer.shap_values(test_data)

    # Reshape SHAP values back to (n_samples, n_features)
    shap_values_reshaped = shap_values[0].reshape(-1, n_features)

    # Visualize feature contribution distributions
    feature_contribution_distribution(shap_values_reshaped, feature_names)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the trained model.')
    parser.add_argument('features', help='Path to the dataset file containing features and labels.')
    args = parser.parse_args()

    # Load the dataset
    x_valid = np.load('x_valid.npy')  # Adjust this path if necessary
    print("Loading model for SHAP explanations...")
    model = tf.keras.models.load_model(args.model)

    explain_with_shap_violin(model, x_valid, FEATURES, SEQUENCE_LENGTH)

if __name__ == "__main__":
    main()
