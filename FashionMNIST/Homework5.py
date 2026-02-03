import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import pandas as pd

fashion_mnist = tf.keras.datasets.mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize the data and reshape for CNN
X_train = X_train.reshape(60000, 28, 28, 1) 
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Define baseline model
def baseline_fc_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define CNN model with Dropout
def cnn_dropout_model():
    model = models.Sequential([
        # First Conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Second Conv block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train baseline model
baseline_model = baseline_fc_model()
baseline_history = baseline_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=128, verbose=1)

# Train CNN with Dropout model
cnn_model = cnn_dropout_model()
cnn_history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=128, verbose=1)

# Plot accuracy for both models
plt.plot(baseline_history.history['accuracy'], label='FC Train')
plt.plot(baseline_history.history['val_accuracy'], label='FC Test')
plt.plot(cnn_history.history['accuracy'], label='CNN Train')
plt.plot(cnn_history.history['val_accuracy'], label='CNN Test')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig("plot.png")

# summary for the CNN model
cnn_model.summary()

# Compare results in a table
results = pd.DataFrame({
    'Model Type': ['Fully Connected (64x64)', 'CNN with Dropout'],
    'Train Accuracy': [baseline_history.history['accuracy'][-1], cnn_history.history['accuracy'][-1]],
    'Test Accuracy': [baseline_history.history['val_accuracy'][-1], cnn_history.history['val_accuracy'][-1]]
})

print(results)
