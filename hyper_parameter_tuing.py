from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from scikeras.wrappers import KerasRegressor
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

def build_model(n_hidden=1, n_neurons=30, learning_rate=0.001):

    # IMPORTANT FIX (type casting)
    n_hidden = int(n_hidden)
    n_neurons = int(n_neurons)
    learning_rate = float(learning_rate)

    input_layer = keras.layers.Input(shape=X_train.shape[1:])
    x = input_layer

    # hidden layers
    for _ in range(n_hidden):
        x = keras.layers.Dense(n_neurons, activation="relu")(x)

    # skip connection
    concat = keras.layers.Concatenate()([input_layer, x])

    output = keras.layers.Dense(1)(concat)

    model = keras.models.Model(inputs=[input_layer], outputs=[output])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(loss="mse", optimizer=optimizer)

    return model

keras_reg = KerasRegressor(
    model=build_model,
    epochs=20,
    batch_size=32,
    verbose=0
)

param_distribs = {
    "model__n_hidden": [1, 2, 3],
    "model__n_neurons": np.arange(10, 100),
    "model__learning_rate": reciprocal(3e-4, 3e-2),
}


# RandomizedSearchCV to find the best hyperparameters 

rnd_search_cv = RandomizedSearchCV(
    keras_reg,
    param_distribs,
    n_iter=10,
    cv=3,
    verbose=2
)

rnd_search_cv.fit(
    X_train_scaled, y_train,
    validation_data=(X_valid_scaled, y_valid),
    callbacks=[keras.callbacks.EarlyStopping(patience=5)]
)

print("Best Hyperparameters:", rnd_search_cv.best_params_)

print("Best Cross-Validation Score:", rnd_search_cv.best_score_)


# Best Hyperparameters: {'model__learning_rate': np.float64(0.004436495390109614), 'model__n_hidden': 3, 'model__n_neurons': np.int64(29)}
#Best Cross-Validation Score: 0.7730453324968084