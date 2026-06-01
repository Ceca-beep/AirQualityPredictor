# Air Quality Prediction with Neural Networks


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


data = pd.read_csv("city_day.csv")

print("=== Dataset Preview ===")
print(data.head())        
print("\nShape:", data.shape) 

    
print("\n=== Missing Values Per Column ===")
print(data.isnull().sum())


data.dropna(axis=0, inplace=True)

print(f"\nRows after removing nulls: {len(data)}")



data['Date'] = pd.to_datetime(data['Date'])


plt.figure(figsize=(10, 5))
data.boxplot(column='AQI', by='City', figsize=(14, 6))
plt.title("AQI Distribution by City")
plt.suptitle("")  # Remove the automatic pandas title
plt.xlabel("City")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_aqi_by_city.png", dpi=150)
plt.close()
print("Saved: eda_aqi_by_city.png")


feature_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3',
                   'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
corr = data[feature_columns].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(feature_columns)))
ax.set_yticks(range(len(feature_columns)))
ax.set_xticklabels(feature_columns, rotation=45, ha='right')
ax.set_yticklabels(feature_columns)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.close()
print("Saved: eda_correlation.png")


input_features = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
                  'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

X = data[input_features]
y = data['AQI']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled  = scaler.transform(X_test)       

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(input_features),)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)   # No activation — regression output
])


model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()  


print("\n=== Training the Model ===")
history = model.fit(
    X_train_scaled, y_train,
    epochs=150,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)


plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],     label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Model Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.close()
print("Saved: training_loss.png")

test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
test_rmse = np.sqrt(test_loss)  # RMSE is easier to interpret (same unit as AQI)

print(f"\n=== Evaluation Results ===")
print(f"Test MSE:  {test_loss:.2f}")
print(f"Test RMSE: {test_rmse:.2f}  (average error in AQI units)")

# PREDICT ON NEW USER INPUT
# AQI scale: 0–50 Good | 51–100 Moderate | 101–200 Unhealthy | 201–300 Very Unhealthy

user_input = pd.DataFrame({
    'PM2.5':   [81],
    'PM10':    [124],
    'NO':      [1.44],
    'NO2':     [20],
    'NOx':     [12],
    'NH3':     [10],
    'CO':      [0.1],
    'SO2':     [15],
    'O3':      [127],
    'Benzene': [0.20],
    'Toluene': [6],
    'Xylene':  [0.06]
})

user_input_scaled = scaler.transform(user_input)
predicted_aqi = model.predict(user_input_scaled, verbose=0)[0][0]

print(f"\n=== Prediction for Sample Input ===")
print(f"Predicted AQI: {predicted_aqi:.1f}")

# Interpret the result
if predicted_aqi <= 50:
    category = "Good"
elif predicted_aqi <= 100:
    category = "Moderate"
elif predicted_aqi <= 200:
    category = "Unhealthy for Sensitive Groups / Unhealthy"
elif predicted_aqi <= 300:
    category = "Very Unhealthy"
else:
    category = "Hazardous"

print(f"AQI Category: {category}")

# Save the model for later use
model.save('air_quality_model.keras')
print("\nModel saved as air_quality_model.keras")