from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# -------------------------
# Load Dataset
# -------------------------
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42
)

# -------------------------
# Feature Scaling
# -------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Build Functional API Model
# -------------------------
input_layer = keras.layers.Input(shape=X_train.shape[1:])

hidden1 = keras.layers.Dense(30, activation="relu")(input_layer)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)

# Skip connection (concatenate input with hidden layer)
concat = keras.layers.Concatenate()([input_layer, hidden2])

output = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs=[input_layer], outputs=[output])

# -------------------------
# Compile Model
# -------------------------
model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mse"]
)

# -------------------------
# Train Model
# -------------------------
history = model.fit(
    X_train_scaled, y_train,
    epochs=20,
    validation_data=(X_valid_scaled, y_valid)
)

# -------------------------
# Evaluate Model
# -------------------------
mse_test = model.evaluate(X_test_scaled, y_test)

print("Test MSE:", mse_test)

# -------------------------
# Model Summary
# -------------------------
model.summary()