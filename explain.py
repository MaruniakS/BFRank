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

def visualize_shap_feature_impact(shap_values, feature_names, classes):
    """
    Visualize SHAP values as feature impact charts for aggregated and per-class results.

    Parameters:
        shap_values: SHAP values, shape (n_classes, n_samples, sequence_length, n_features).
        feature_names: List of feature names.
        classes: List of class names or indices (e.g., [0, 1, 2]).
    """
    shap_values = np.array(shap_values)

    # Aggregated SHAP values
    aggregated_shap_values = np.mean(np.abs(shap_values), axis=(0, 1, 2))

    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, aggregated_shap_values)
    plt.title("Aggregated Feature Impact (All Classes)")
    plt.xlabel("Features")
    plt.ylabel("Mean Absolute SHAP Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.show()

    # Per-class SHAP visualization
    for class_idx, class_name in enumerate(classes):
        per_class_shap_values = np.mean(np.abs(shap_values[class_idx]), axis=(0, 1))

        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, per_class_shap_values)
        plt.title(f"Feature Impact for Class {class_name}")
        plt.xlabel("Features")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y")
        plt.show()


def shap_waterfall_plot(explainer, test_sample, feature_names, sequence_length, n_features):
    """
    Visualize SHAP values for a single sample as a waterfall plot.

    Parameters:
        explainer: SHAP explainer object.
        test_sample: A single test sample to explain (shape: 1 x flattened_length).
        feature_names: List of feature names.
        sequence_length: Number of time steps.
        n_features: Number of features per time step.
    """
    # Compute SHAP values for the single test sample
    shap_values = explainer.shap_values(test_sample)
    
    # Reshape SHAP values back to the original format
    reshaped_shap_values = shap_values[0].reshape(sequence_length, n_features)

    # Aggregate SHAP values across time steps for visualization
    aggregated_shap_values = np.mean(np.abs(reshaped_shap_values), axis=0)

    # Plot waterfall chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_names, aggregated_shap_values)
    plt.title("Waterfall Plot: Feature Impact on Prediction")
    plt.xlabel("Features")
    plt.ylabel("Mean Absolute SHAP Value (Aggregated Across Time)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.show()


def shap_summary_plot(explainer, test_data, feature_names, sequence_length, n_features):
    """
    Generate a summary plot of SHAP values.

    Parameters:
        explainer: SHAP explainer object.
        test_data: Test data for which to compute SHAP values.
        feature_names: List of feature names.
        sequence_length: Number of time steps.
        n_features: Number of features per time step.
    """
    # Compute SHAP values for the entire test dataset
    shap_values = explainer.shap_values(test_data)
    
    # Reshape SHAP values to their original shape
    shap_values_reshaped = [
        values.reshape(-1, sequence_length, n_features) for values in shap_values
    ]

    # Aggregate SHAP values across all time steps
    shap_values_aggregated = np.mean(np.abs(shap_values_reshaped), axis=2)  # Shape: (n_classes, n_samples, n_features)

    # Aggregate SHAP values across all samples
    shap_values_final = np.mean(shap_values_aggregated, axis=1)  # Shape: (n_classes, n_features)

    # Generate summary plots for each class
    for class_idx, shap_vals in enumerate(shap_values_final):
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, shap_vals)
        plt.title(f"SHAP Summary Plot for Class {class_idx}")
        plt.xlabel("Features")
        plt.ylabel("Mean Absolute SHAP Value")
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y")
        plt.show()


def shap_force_plot(explainer, test_sample, feature_names, sequence_length, n_features):
    """
    Visualize SHAP values as a force plot for a single test sample.

    Parameters:
        explainer: SHAP explainer object.
        test_sample: A single test sample to explain (shape: 1 x flattened_length).
        feature_names: List of feature names.
        sequence_length: Number of time steps.
        n_features: Number of features per time step.
    """
    # Compute SHAP values for the single test sample
    shap_values = explainer.shap_values(test_sample)

    # Reshape SHAP values back to the original sequence structure
    shap_values_reshaped = shap_values[0].reshape(sequence_length, n_features)

    # Aggregate SHAP values across time steps
    aggregated_shap_values = np.sum(shap_values_reshaped, axis=0)  # Shape: (n_features,)

    # Create a force plot
    shap.initjs()
    shap.force_plot(
        base_value=explainer.expected_value[0],
        shap_values=aggregated_shap_values,
        feature_names=feature_names,
        matplotlib=True
    )
    plt.title("SHAP Force Plot for a Single Test Sample")
    plt.show()


def explain_with_shap_per_time(model, x_data, feature_names, sequence_length, plot_type):
    """
    Compute and visualize SHAP values for multiple examples.

    Parameters:
        model: Trained TensorFlow model.
        x_data: Input data (n_samples, sequence_length, n_features) as a list or array.
        feature_names: List of feature names.
        sequence_length: Number of time steps.
    """
    n_features = len(feature_names)
    x_data = np.array(x_data)
    x_data_flattened = x_data.reshape(x_data.shape[0], -1)

    background_data = x_data_flattened[:10]
    test_data = x_data_flattened[:100]

    explainer = shap.KernelExplainer(
        lambda x: model.predict(x.reshape(-1, sequence_length, n_features)),
        background_data
    )

    shap_values = explainer.shap_values(test_data)
    shap_values_reshaped = [values.reshape(-1, sequence_length, n_features) for values in shap_values],

    if plot_type == "summary":
        print("Generating summary plot for test samples...")
        shap_summary_plot(
            explainer, 
            test_data, 
            feature_names, 
            sequence_length=sequence_length, 
            n_features=n_features
        )
    elif plot_type == "waterfall":
        for i in range(len(test_data)):
            print(f"Explaining sample {i + 1} with waterfall plot:")
            shap_waterfall_plot(
                explainer, 
                test_data[[i]], 
                feature_names, 
                sequence_length, 
                n_features
            )
    elif plot_type == "force":
        print("Generating force plot for a single test sample...")
        shap_force_plot(
            explainer, 
            test_data[[0]],  # First test sample
            feature_names=feature_names, 
            sequence_length=sequence_length, 
            n_features=n_features
        )
    elif plot_type == "impact":
        visualize_shap_feature_impact(
            shap_values=shap_values_reshaped,
            feature_names=feature_names,
            classes=[0, 1, 2]
        ) 
    else:
        print(f"Invalid plot type: {plot_type}")            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Path to the trained model.')
    parser.add_argument('features', help='Path to the dataset file containing features and labels.')
    parser.add_argument('plot_type', help='Type of SHAP plot to generate: summary, waterfall, force, impact.')
    args = parser.parse_args()

    x_valid = np.load('x_valid.npy')
    print("Loading model for SHAP explanations...")
    model = tf.keras.models.load_model(args.model)

    explain_with_shap_per_time(model, x_valid, FEATURES, SEQUENCE_LENGTH, args.plot_type)

if __name__ == "__main__":
    main()
