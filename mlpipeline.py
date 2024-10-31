import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# App title
st.title("Machine Learning Pipeline Demonstration")

# Model selection
st.sidebar.header("Select Model Type")
model_type = st.sidebar.selectbox("Choose a model type:", ["Classification", "Regression", "Clustering"])

# Step 1: Data Upload
st.header("1. Data Upload")
uploaded_file = st.file_uploader("Upload your CSV data file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(df.head())

    # Step 2: Data Preprocessing
    st.header("2. Data Preprocessing")
    if st.checkbox("Show Data Summary"):
        st.write(df.describe())
    
    # Select Features and Target based on the model type
    features = st.multiselect("Select features for training", options=df.columns)
    target = None
    if model_type in ["Classification", "Regression"]:
        target = st.selectbox("Select target variable", options=df.columns) if features else None
    
    # Data preprocessing steps based on model type
    if model_type == "Classification" and target:
        # Convert target to binary if continuous for classification
        if df[target].dtype in ['float64', 'int64'] and len(df[target].unique()) > 2:
            st.warning("The selected target is continuous. Converting to binary classification by thresholding at the median.")
            threshold = df[target].median()
            df[target] = (df[target] > threshold).astype(int)
        elif len(df[target].unique()) == 1:
            st.error("The selected target has only one unique value and cannot be used for classification.")
            st.stop()

    # Feature Scaling
    if st.checkbox("Apply Standard Scaling") and features:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        st.write("Data after Scaling:")
        st.write(df.head())
    elif not features:
        st.warning("Please select at least one feature for scaling.")

    # Step 3: Model Training and Evaluation
    st.header("3. Model Training and Evaluation")
    if features:
        if model_type == "Classification" and target:
            # Classification algorithm selection
            st.sidebar.header("Select Classification Algorithm")
            classifier_name = st.sidebar.selectbox("Classifier", ["Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes"])

            # Classification pipeline
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

            # Initialize the selected classifier
            if classifier_name == "Logistic Regression":
                model = LogisticRegression()
            elif classifier_name == "K-Nearest Neighbors":
                n_neighbors = st.sidebar.slider("Number of neighbors (K)", 1, 15, 5)
                model = KNeighborsClassifier(n_neighbors=n_neighbors)
            elif classifier_name == "Decision Tree":
                max_depth = st.sidebar.slider("Max depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            elif classifier_name == "Naive Bayes":
                model = GaussianNB()

            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluation
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))
            
            # Confusion Matrix
            st.write("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, 
                        xticklabels=model.classes_, yticklabels=model.classes_)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            st.pyplot(fig)

        elif model_type == "Regression" and target:
            # Regression pipeline
            X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Evaluation
            st.write("Mean Squared Error:", mean_squared_error(y_test, y_pred))
            st.write("R^2 Score:", r2_score(y_test, y_pred))
            
            # Plot true vs predicted values
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
            ax.set_xlabel("True Values")
            ax.set_ylabel("Predictions")
            st.pyplot(fig)

        elif model_type == "Clustering":
            # Clustering pipeline
            n_clusters = st.slider("Select number of clusters", 2, 10, 3)
            model = KMeans(n_clusters=n_clusters, random_state=42)
            df['Cluster'] = model.fit_predict(df[features])

            # Display cluster centers
            st.write("Cluster Centers:")
            st.write(pd.DataFrame(model.cluster_centers_, columns=features))

            # Plot clusters
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=features[0], y=features[1], hue="Cluster", palette="viridis", ax=ax)
            ax.set_title("Clustering Result")
            st.pyplot(fig)

    # Step 4: Make Predictions or Show Cluster Labels
    st.header("4. Make Predictions / View Clustering Results")
    if model_type in ["Classification", "Regression"] and features and target:
        input_data = {feature: st.number_input(f"Input {feature}", value=0.0) for feature in features}
        if st.button("Predict"):
            input_df = pd.DataFrame([input_data])
            if 'scaler' in locals():
                input_df[features] = scaler.transform(input_df[features])
            prediction = model.predict(input_df)
            st.write("Prediction:", prediction[0])
    elif model_type == "Clustering" and features:
        st.write("Cluster labels have been added to the dataset.")
        st.write(df[['Cluster'] + features].head())
