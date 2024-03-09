import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from keras.layers import TextVectorization
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding


pd.set_option("display.max_columns", None)

raw_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

raw_df.info()
raw_df.sample(10)

raw_df["comment_text"].values[0]
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for col in target_cols:
    print(raw_df[col].value_counts())


##################
# Prepare dataset
##################

X = raw_df["comment_text"]
y = raw_df[raw_df.columns[2:]].values


# Calculate sequence lengths
def calculate_sequence_lengths(data):
    sequence_lengths = []
    for text in data["comment_text"]:
        sequence_length = len(text.split())  # Split text and count tokens
        sequence_lengths.append(sequence_length)
    return sequence_lengths


sequence_lengths = calculate_sequence_lengths(raw_df)
sum(sequence_lengths) / len(sequence_lengths)

# Visualize the distribution
plt.hist(sequence_lengths, bins=50)
plt.xlabel("Sequence Length")
plt.ylabel("Frequency")
plt.title("Distribution of Sequence Lengths")
plt.show()

# Visualize the distribution using a KDE plot
sns.histplot(sequence_lengths, kde=True)
plt.xlabel("Sequence Length")
plt.ylabel("Density")
plt.title("Distribution of Sequence Lengths")
plt.show()

OUTPUT_SEQ_LEN = 150

# Calculate the number of unique words in the dataset
unique_words = set()
for text in raw_df["comment_text"]:
    unique_words.update(text.lower().split())

# Choose max_features based on a percentage of unique words
percentage = 0.50  # Adjust as needed
MAX_FEATURES = int(len(unique_words) * percentage)

# Print the chosen value
print("Chosen max_features:", MAX_FEATURES)


vectorizer = TextVectorization(
    max_tokens=MAX_FEATURES, output_sequence_length=OUTPUT_SEQ_LEN, output_mode="int"
)
vectorizer.adapt(X.values)
vectorized_text = vectorizer(X.values)

# MCSHBAP - map, chache, shuffle, batch, prefetch  from_tensor_slices, list_file
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache()
dataset = dataset.shuffle(160000)
dataset = dataset.batch(16)
dataset = dataset.prefetch(8)


# Shuffle the dataset before splitting
shuffled_dataset = dataset.shuffle(len(dataset), seed=42)

# Split the dataset into training, validation, and test sets
train_size = int(0.7 * len(shuffled_dataset))
val_size = int(0.2 * len(shuffled_dataset))
test_size = len(shuffled_dataset) - train_size - val_size

train_dataset = shuffled_dataset.take(train_size)
val_dataset = shuffled_dataset.skip(train_size).take(val_size)
test_dataset = shuffled_dataset.skip(train_size + val_size).take(test_size)


#############
# Modelling
#############

from keras.callbacks import ModelCheckpoint

# Define the ModelCheckpoint callback
model_checkpoint_callback = ModelCheckpoint(
    filepath="finalmodeltoxic.h5", monitor="val_loss", mode="min", save_best_only=True
)


model = Sequential(
    [
        Embedding(MAX_FEATURES + 1, 32),
        Bidirectional(LSTM(32, activation="tanh")),
        Dense(128, activation="relu"),
        Dense(256, activation="relu"),
        Dense(128, activation="relu"),
        Dense(6, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
model.summary()

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[model_checkpoint_callback],
)


###################
# Visualize the training
###################

# Set the style
plt.style.use("dark_background")


# Function to plot training and validation loss curves
def plot_loss_curves(history, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Training Loss", color="orange")
    plt.plot(history.history["val_loss"], label="Validation Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    if save_path:
        plt.savefig(save_path + "_white_loss.png", bbox_inches="tight")
    plt.show()


# Function to plot training and validation accuracy curves
def plot_accuracy_curves(history, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Training Accuracy", color="orange")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curves")
    plt.legend()
    if save_path:
        plt.savefig(save_path + "_white_accuracy.png", bbox_inches="tight")
    plt.show()


plot_loss_curves(history)
plot_accuracy_curves(history)

# Save and load the model
model.save("finaltoxicmodel.h5")
loaded_model = tf.keras.models.load_model("finaltoxicmodel.h5")

# Evaluate
test_loss, test_accuracy = loaded_model.evaluate(test_dataset)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Make predictions on the test data.
# Preprocess the text data in the test dataframe
test_text = test_df["comment_text"]

# Tokenize and vectorize the test text using the same vectorizer used for training data
vectorized_test_text = vectorizer(test_text)

# Create a test dataset from the vectorized test text
test_dataset = tf.data.Dataset.from_tensor_slices(vectorized_test_text)
test_dataset = test_dataset.batch(16)

predictions = loaded_model.predict(test_dataset)
predictions.shape

test_probs = tf.sigmoid(predictions).numpy()
test_df.head()

submission_df = pd.read_csv("data/sample_submission.csv")
submission_df.head()


submission_df[target_cols] = test_probs
submission_df.to_csv("data/toxic_submission_df.csv", index=None)

##########################
# Custom Predictions
##########################
import pickle

# Pickle the config and weights
pickle.dump(
    {"config": vectorizer.get_config(), "weights": vectorizer.get_weights()},
    open("tv_layer.pkl", "wb"),
)

loaded_toxic_model = tf.keras.models.load_model("finaltoxicmodel.h5")

from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_v = TextVectorization.from_config(from_disk["config"])
# You have to call `adapt` with some dummy data (BUG in Keras)
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk["weights"])


def toxicity_classifier(comment):
    vectorized_comment = new_v([comment])
    results = loaded_toxic_model.predict(vectorized_comment)
    tox = [
        "Toxicity",
        "Severe toxicity",
        "Obscene",
        "Threat",
        "Insult",
        "Identity hate",
    ]
    tox_dict = {}
    for idx, col in enumerate(tox):
        confidence_percentage = round(results[0][idx] * 100, 2)
        # Convert numpy.bool_ to native Python bool
        is_toxic = bool(results[0][idx] > 0.45)
        tox_dict[col] = [is_toxic, confidence_percentage]

    return tox_dict

tox_dict = toxicity_classifier("Hello friend")