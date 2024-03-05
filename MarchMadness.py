from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Assume data is a DataFrame with the features and target is the column to predict
data = ...
target = ...

# Split the data into a training set and a test set
train_data, test_data, train_target, test_target = train_test_split(data, target)

# Create a model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_data, train_target, epochs=10)

# Evaluate the model
loss = model.evaluate(test_data, test_target)

# Make predictions
predictions = model.predict(new_data)


# Assume teams is a list of team names and probabilities is a list of predicted probabilities
teams = ...
probabilities = ...

plt.barh(teams, probabilities)
plt.xlabel('Probability of Winning')
plt.title('Predicted NCAA Tournament Outcomes')
plt.show()

# Save the model
model.save('ncaa_model.keras')


# Assume y_true are the true outcomes and y_pred are the predicted outcomes
y_true = ...
y_pred = ...

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Log Loss:", log_loss(y_true, y_pred))
print("AUC-ROC:", roc_auc_score(y_true, y_pred))
