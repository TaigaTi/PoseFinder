import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("pose_dataset.csv")

# Split features and labels
X = df.iloc[:, :-1].values # keypoints
y = df.iloc[:, -1].values # labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# One-hot encode labels for training 
y_train_cat = to_categorical(y_train, num_classes=2)
y_test_cat = to_categorical(y_test, num_classes=2)

# Build model
model = Sequential([
    Dense(128, activation='relu', input_shape=(66,)),
    Dense(64, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train_cat, epochs=20, batch_size=16, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat)
print(f"\nTest Accuracy: {acc:.2f}")

# Classification report
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
print("\nClassification Report:\n", classification_report(y_test, y_pred_labels))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Training Accuracy")
plt.legend()
plt.show()