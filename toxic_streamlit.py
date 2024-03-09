import streamlit as st
import pandas as pd

st.set_page_config(layout="wide")
st.markdown(
    """<h1 style='color: #3498db; text-align: center;'>️Toxic Comment Classifier</h1>""",
    unsafe_allow_html=True,
)

# HOME
tab_home, tab_data, model_tab, make_preds = st.tabs(
    [
        "Home",
        "Data",
        "Model Development Overview",
        "Classify Text!",
    ]
)

tab_home.subheader("About the Kaggle Competition")

tab_home.write(
    """Discussing things you care about can be difficult online due to the threat of abuse and harassment, leading to limited or closed user comments. The Conversation AI team, a research initiative by Jigsaw and Google, aims to improve online conversation, focusing on negative behaviors like toxic comments. In this competition, participants were challenged to build models capable of detecting various types of toxicity better than existing ones, using a dataset of Wikipedia comments."""
)

tab_home.write(
    """For more information, you can visit the Kaggle competition page [here](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview)."""
)


tab_home.subheader("Objective📝")

objectives = [
    "Develop a machine learning model to classify toxic comments into categories.",
    "Utilize NLP techniques to preprocess and vectorize text data.",
    "Train a deep learning model with LSTM layers to capture text patterns.",
    "Create a Streamlit app for interactive demonstration and prediction of toxic comments.",
]

for objective in objectives:
    tab_home.markdown(f"- {objective}")

# DATA
tab_data.subheader("Data Overview 📊")

tab_data.write("### Data Source")
tab_data.write(
    "[Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)"
)

tab_data.markdown(
    """
    You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. Each comment in the dataset has been labeled for various types of toxicity, including toxic, severe_toxic, obscene, threat, insult, and identity_hate. The task is to create a machine learning model that predicts the probability of each type of toxicity for each comment.

    The provided dataset consists of comments from Wikipedia's talk page edits, where human raters have annotated the comments based on their perceived toxicity. Each comment can have multiple labels corresponding to different types of toxicity. For example, a comment may be labeled as toxic and obscene simultaneously if it contains both types of behavior.

    The types of toxicity labels in the dataset are as follows:
    - toxic
    - severe_toxic
    - obscene
    - threat
    - insult
    - identity_hate

    The objective is to build a predictive model that accurately identifies and classifies these different types of toxic behaviors in comments, enabling platforms to better manage and moderate online discussions to promote respectful and productive conversations.
    """
)

## Model
model_tab.subheader("Model Development and Evaluation  🤖📊")

with model_tab.expander("Model Development Overview"):
    model_tab.write(
        """
        In the process of developing the toxic comment classifier model, several key steps were undertaken to ensure its effectiveness and accuracy in identifying different types of toxicity in online comments.

        1. **Data Preparation:** The first step involved preprocessing the comment text data, including tokenization and vectorization using techniques such as TextVectorization. Additionally, the dataset was split into training, validation, and test sets to facilitate model training and evaluation.

        2. **Model Architecture:** The model architecture was designed to effectively capture the sequential patterns in the text data. A deep learning architecture consisting of embedding layers, bidirectional LSTM layers, and dense layers was implemented. This architecture enables the model to learn complex relationships and patterns in the comment text for toxicity classification.

        3. **Model Training:** The model was trained using the training dataset and evaluated on the validation set to monitor its performance and prevent overfitting. The training process involved optimizing the model's parameters using the binary cross-entropy loss function and the Adam optimizer.

        4. **Evaluation Metrics:** The model's performance was evaluated using metrics such as accuracy and loss on both the training and validation sets. Additionally, visualizations such as loss curves and accuracy curves were generated to assess the model's convergence and generalization capability.

        5. **Model Evaluation:** After training, the model's performance was further evaluated on an independent test set to assess its real-world effectiveness in classifying toxic comments. The evaluation included calculating metrics such as test loss and test accuracy to measure the model's overall performance.

        6. **Model Deployment:** Once the model was trained and evaluated, it was saved and deployed for use in the Streamlit application. Users can interact with the deployed model to predict the probability of different types of toxicity in custom input comments, facilitating real-time toxicity classification in online discussions.
        """
    )


# Make Preds
make_preds.subheader("Classify Text")


import tensorflow as tf
import pickle

# Load the toxic comment classifier model and text vectorization layer
loaded_toxic_model = tf.keras.models.load_model("finaltoxicmodel.h5")
from_disk = pickle.load(open("tv_layer.pkl", "rb"))
new_v = tf.keras.layers.TextVectorization.from_config(from_disk["config"])
new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
new_v.set_weights(from_disk["weights"])


# Define toxicity classifier function
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
        tox_dict[col] = {
            "is_toxic": is_toxic,
            "confidence_percentage": confidence_percentage,
        }

    return tox_dict


# Text input field for user to input comment
user_comment = make_preds.text_input("Enter your comment here:")

# Button to trigger prediction
if make_preds.button("Predict"):
    # Check if user has entered text
    if user_comment:
        # Call toxicity classifier function to make prediction
        prediction_result = toxicity_classifier(user_comment)

        # Display prediction result
        make_preds.write(
            "<span style='color: green;'>Prediction Result:</span>",
            unsafe_allow_html=True,
        )
        for label, values in prediction_result.items():
            make_preds.write(
                f"{label}: {'Toxic' if values['is_toxic'] else 'Not Toxic'} - Confidence: {values['confidence_percentage']}%"
            )
    else:
        make_preds.write("Please enter a comment to classify.")
