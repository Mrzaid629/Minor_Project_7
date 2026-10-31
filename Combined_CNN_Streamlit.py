import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------
# PROJECT: CNN IMAGE CLASSIFICATION AND STREAMLIT APPLICATION
# Part A: MNIST handwritten digit classification using CNN
# Part B: Student marks calculator using Streamlit
# ---------------------------------------------------------

st.set_page_config(
    page_title="CNN MNIST and Student Marks Calculator",
    page_icon="🎓",
    layout="wide"
)

st.title("CNN Image Classification and Streamlit Application")
st.write(
    "This project contains two parts: handwritten digit classification "
    "using a Convolutional Neural Network (CNN), and a student marks calculator."
)

# ---------------------- PART A: CNN ----------------------

@st.cache_resource(show_spinner=False)
def train_cnn_model():
    """Load MNIST, preprocess it, build a CNN, train, and evaluate it."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values and add the channel dimension.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build the CNN model.
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    predictions = model.predict(x_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)

    return (
        model, history, x_test, y_test, y_pred,
        float(test_loss), float(test_accuracy)
    )


cnn_tab, marks_tab = st.tabs(["Part A: CNN Image Classification", "Part B: Student Marks Calculator"])

with cnn_tab:
    st.header("MNIST Handwritten Digit Classification")
    st.write(
        "The MNIST dataset contains 28 × 28 grayscale images of handwritten "
        "digits from 0 to 9. The CNN learns patterns in these images and predicts the digit."
    )

    st.info(
        "Select the button below to load MNIST, preprocess the images, build the CNN, "
        "train it for 5 epochs, and evaluate it. The first run may take a few minutes."
    )

    if st.button("Train and Evaluate CNN", type="primary"):
        with st.spinner("Training the CNN model. Please wait..."):
            results = train_cnn_model()
        st.session_state["cnn_results"] = results

    if "cnn_results" in st.session_state:
        model, history, x_test, y_test, y_pred, test_loss, test_accuracy = st.session_state["cnn_results"]

        st.subheader("Dataset details")
        st.write("Training images: 60,000")
        st.write("Testing images: 10,000")
        st.write("Image shape: 28 × 28 pixels (grayscale)")
        st.write("Classes: digits 0 to 9")

        st.subheader("CNN model summary")
        summary_lines = []
        model.summary(print_fn=lambda line: summary_lines.append(line))
        st.code("\n".join(summary_lines))

        st.subheader("Training and validation graphs")
        col1, col2 = st.columns(2)

        with col1:
            fig_accuracy, ax_accuracy = plt.subplots()
            ax_accuracy.plot(history.history["accuracy"], label="Training Accuracy")
            ax_accuracy.plot(history.history["val_accuracy"], label="Validation Accuracy")
            ax_accuracy.set_title("Training and Validation Accuracy")
            ax_accuracy.set_xlabel("Epoch")
            ax_accuracy.set_ylabel("Accuracy")
            ax_accuracy.legend()
            st.pyplot(fig_accuracy)
            plt.close(fig_accuracy)

        with col2:
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(history.history["loss"], label="Training Loss")
            ax_loss.plot(history.history["val_loss"], label="Validation Loss")
            ax_loss.set_title("Training and Validation Loss")
            ax_loss.set_xlabel("Epoch")
            ax_loss.set_ylabel("Loss")
            ax_loss.legend()
            st.pyplot(fig_loss)
            plt.close(fig_loss)

        st.subheader("Test performance")
        final_train_accuracy = history.history["accuracy"][-1] * 100
        final_val_accuracy = history.history["val_accuracy"][-1] * 100
        correct = int(np.sum(y_test == y_pred))
        incorrect = int(np.sum(y_test != y_pred))

        metric1, metric2, metric3, metric4 = st.columns(4)
        metric1.metric("Test Accuracy", f"{test_accuracy * 100:.2f}%")
        metric2.metric("Test Loss", f"{test_loss:.4f}")
        metric3.metric("Training Accuracy", f"{final_train_accuracy:.2f}%")
        metric4.metric("Validation Accuracy", f"{final_val_accuracy:.2f}%")

        st.write(f"Correct predictions: **{correct}**")
        st.write(f"Incorrect predictions: **{incorrect}**")
        st.write(f"Total test images: **{len(y_test)}**")

        st.subheader("Actual and predicted examples")
        fig_examples, axes = plt.subplots(3, 5, figsize=(12, 7))
        for index, ax in enumerate(axes.flat):
            ax.imshow(x_test[index].squeeze(), cmap="gray")
            ax.set_title(f"Actual: {y_test[index]}\nPredicted: {y_pred[index]}")
            ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig_examples)
        plt.close(fig_examples)

        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=range(10)
        )
        display.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        st.pyplot(fig_cm)
        plt.close(fig_cm)

        st.success("CNN training and evaluation completed.")

    else:
        st.caption("CNN results will appear here after you click Train and Evaluate CNN.")

# ---------------- PART B: STREAMLIT MARKS CALCULATOR ----------------

with marks_tab:
    st.header("🎓 Student Marks Calculator")
    st.write("Enter student details and marks for the five subjects.")

    student_name = st.text_input("Student Name", key="student_name")

    subjects = [
        "English",
        "Mathematics",
        "Science",
        "Computer Science",
        "Social Studies"
    ]

    marks = []
    mark_columns = st.columns(2)
    for index, subject in enumerate(subjects):
        with mark_columns[index % 2]:
            mark = st.number_input(
                subject,
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                key=f"mark_{index}"
            )
            marks.append(mark)

    if st.button("Calculate Results"):
        if not student_name.strip():
            st.error("Please enter student name.")
        else:
            total = sum(marks)
            percentage = (total / 500) * 100

            if min(marks) < 35:
                grade = "F"
                result = "FAIL"
            elif percentage >= 90:
                grade = "A+"
                result = "PASS"
            elif percentage >= 80:
                grade = "A"
                result = "PASS"
            elif percentage >= 70:
                grade = "B"
                result = "PASS"
            elif percentage >= 60:
                grade = "C"
                result = "PASS"
            else:
                grade = "D"
                result = "PASS"

            st.subheader("Student Results")
            st.write("Student Name:", student_name.strip())
            st.write("Total Marks:", total, "/ 500")
            st.write("Percentage:", f"{percentage:.2f}%")
            st.write("Grade:", grade)

            if result == "PASS":
                st.success("Result: PASS")
            else:
                st.error("Result: FAIL")

st.divider()
st.caption("Project: CNN Image Classification and Streamlit Application")
