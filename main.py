import shap
import tensorflow as tf
import numpy as np

# Simple dummy model for testing
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),  # 5 features
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100), epochs=1, batch_size=10)

# SHAP test data
background_data = np.random.rand(10, 5)
test_data = np.random.rand(5, 5)

# SHAP explainer
explainer = shap.KernelExplainer(model.predict, background_data)
shap_values = explainer.shap_values(test_data)

# Summary plot
shap.summary_plot(shap_values, test_data)
