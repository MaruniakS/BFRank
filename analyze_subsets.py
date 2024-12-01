import shap
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from constants import FEATURES, SEQUENCE_LENGTH

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

def compute_shap_values(model, x_data, feature_names, sequence_length):
    """
    Compute SHAP values for the provided dataset.

    Parameters:
        model: Trained TensorFlow model.
        x_data: Input data (n_samples, sequence_length, n_features).
        feature_names: List of feature names.
        sequence_length: Number of time steps.

    Returns:
        SHAP values reshaped to (n_classes, n_samples, n_features).
    """
    n_features = len(feature_names)
    x_data = np.array(x_data)
    x_data_flattened = x_data.reshape(x_data.shape[0], -1)

    # Use the first 10 samples as background and the entire dataset as test data
    background_data = x_data_flattened[:10]
    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, sequence_length, n_features)),
        background_data
    )
    shap_values = explainer.shap_values(x_data_flattened)

    # Reshape SHAP values back to (n_classes, n_samples, n_features)
    shap_values_reshaped = [
        values.reshape(-1, sequence_length, n_features).mean(axis=1) for values in shap_values
    ]
    return np.array(shap_values_reshaped)  # Shape: (n_classes, n_samples, n_features)


def analyze_feature_impact_across_random_subsets(shap_values, feature_names, subset_size=100, n_subsets=5):
    """
    Analyze and compare the global impact of features across multiple random subsets.

    Parameters:
        shap_values: SHAP values, shape (n_classes, n_samples, n_features).
        feature_names: List of feature names.
        subset_size: Number of samples per subset.
        n_subsets: Number of subsets to generate and compare.
    """
    import random

    shap_values = np.array(shap_values)  # Ensure SHAP values are a NumPy array
    n_classes, n_samples, n_features = shap_values.shape
    subset_results = {}

    for i in range(n_subsets):
        # Select a random subset of indices
        random_indices = random.sample(range(n_samples), subset_size)
        subset_shap = shap_values[:, random_indices, :]  # Select SHAP values for the subset

        # Compute the mean absolute SHAP value for each feature
        subset_impact = np.mean(np.abs(subset_shap), axis=(0, 1))  # Shape: (n_features,)
        subset_results[f"Subset {i+1}"] = subset_impact

    # Create a DataFrame for visualization
    subset_impact_df = pd.DataFrame(subset_results, index=feature_names)

    # Plot the results
    subset_impact_df.plot(kind="bar", figsize=(14, 8), width=0.8)
    plt.title("Feature Impact Across Random Subsets")
    plt.xlabel("Features")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Subsets")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the trained model.')
    parser.add_argument('features', help='Path to the dataset file containing features and labels.')
    args = parser.parse_args()

    # Load dataset
    x_valid = np.load('x_valid.npy')
    print(f"Loaded dataset with shape: {x_valid.shape}")

    # Load model
    print("Loading model...")
    model = tf.keras.models.load_model(args.model)

    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = compute_shap_values(model, x_valid, FEATURES, SEQUENCE_LENGTH)

    # Analyze feature impact across random subsets
    print("Analyzing feature impact across random subsets...")
    analyze_feature_impact_across_random_subsets(
        shap_values=shap_values,
        feature_names=FEATURES,
        subset_size=100,
        n_subsets=5
    )


if __name__ == "__main__":
    main()
