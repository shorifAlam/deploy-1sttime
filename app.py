import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay

#surpress error
st.set_option('deprecation.showPyplotGlobalUse', False)

#set page title and icon
st.set_page_config(page_title="IRIS Dataset", page_icon= ":pushpin:")
page = st.sidebar.selectbox("Select a page",["Home","Data Overview", "EDA", "Modeling", "Make Predictions!", "Extras"])

df = pd.read_csv("data/iris.csv")
# Build a homepage
if page == "Home":
    st.title("IRIS dataset explorer app")
    st.subheader("Welcome to IRIS dataset explorer app")
    st.write("This app is design to make the exploration and analysis if the IRIS dataset easy and accessible to all!")
    st.image("https://bouqs.com/blog/wp-content/uploads/2021/11/iris-flower-meaning-and-symbolism.jpg")



#Build Data overview page
if page == "Data Overview":
    st.title(":1234: Data Overview")
    st.subheader("About the data")
    st.write("this is one the earliest datasets used in the literature on classification and is widely used in statistics and machine learning.")
    st.image("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Machine+Learning+R/iris-machinelearning.png")
    st.link_button("Click here to learn more","https://en.wikipedia.org/wiki/Iris_(plant)", help = "Click me", type = "primary")
    st.subheader("Quick Glance at the data.")

    # Display Dataset
    if st.checkbox("DataFrame"):
        st.dataframe(df)
        st.balloons()
    # column list
    if st.checkbox("Column List"):
        st.code(f"Colums: {df.columns.tolist()}")

        if st.toggle("Further breakdown of columns"):
            num_cols = df.select_dtypes(include = "number").columns.tolist()
            obj_cols = df.select_dtypes(include = "object").columns.tolist()
            st.code(f"Numerical Columns: {num_cols} \nObject Columns: {obj_cols}")
    if st.checkbox("Shape"):
        st.write(f"There are {df.shape[0]} rows and {df.shape[1]} columns.")


# Build EDA page
if page == "EDA":
    st.title(":bar_chart: EDA")
    num_cols = df.select_dtypes(include = "number").columns.tolist()
    obj_cols = df.select_dtypes(include = "object").columns.tolist()
    eda_type = st.multiselect("What type of EDA are you interested in exploring?",
                              ["Histogram","Box Plots", "Scatterplots", "Count Plots"])
   
   # st.write(eda_type)
    # HISTOGRAM:
    if "Histogram" in eda_type:
        st.subheader("Histogram - Visualizing Numerical Distributions")
        h_selected_cal = st.selectbox("Select a numerical column for histogram:", num_cols, index = None)

        if h_selected_cal: 
            chart_title = f'Distribution of {" ".join(h_selected_cal.split("_")).title()}'
            if st.toggle("Species Hue on Histogram"):
                st.plotly_chart(px.histogram(df, x = h_selected_cal, title = chart_title, color = "species", barmode = "overlay"))
            else:
                st.plotly_chart(px.histogram(df, x = h_selected_cal, title = chart_title))

    # BOXPLOT 
    if "Box Plots" in eda_type:
        st.subheader("Box Plots - Visualizing Numerical Column")
        b_selected_cal = st.selectbox("Select a numerical column for box plots:", num_cols, index = None)    
        if b_selected_cal:
            chart_title = f'Distribution of {" ".join(b_selected_cal.split("_")).title()}'
            if st.toggle("Species Hue on Box Plot"):
                st.plotly_chart(px.box(df, x = b_selected_cal, y = "species", title = chart_title, color = "species"))
            else:
                st.plotly_chart(px.box(df, x = b_selected_cal, title = chart_title))
                
# Build Modeling page
if page == "Modeling":
    st.title(":gear: Modeling")
    st.markdown("On this page, you can see how well different machine models make predictions on this ***IRIS*** species!")

    # set up x and y
    features = ["sepal_length", "sepal_width", "petal_length", "petal_length"]
    X = df[features]
    y = df["species"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    model_option = st.selectbox("select a model", ["KNN", "Logistic Regression", "Rendom Forest"], index = None)

    if model_option:
        st.write(f"you selected {model_option}")

        if model_option == "KNN":
            k_value = st.slider("select the number of neighbors (K)", 3, 29, 3, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        
        elif model_option == "Logistic Regression":
            model = LogisticRegression()

        elif model_option == "Rendom Forest":
            model = RandomForestClassifier()
        
        if st.button("Lets see the performance"):
            model.fit(X_train, y_train)

            # Display results
            st.subheader(f"{model} Evaluation")
            st.text(f"Training Accuracy: {round(model.score(X_train, y_train)*100, 2)}%")
            st.text(f"Testing Accuracy: {round(model.score(X_test, y_test)*100, 2)}%")

            st.subheader("confusion matrix: ")
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap = "Blues")
            st.pyplot()


# Predictions Page
if page == "Make Predictions!":
    st.title(":rocket: Make Predictions on Iris Dataset")

    # Create sliders for user to input data
    st.subheader("Adjust the sliders to input data:")

    s_l = st.slider("Sepal Length (cm)", 0.0, 10.0, 0.0, 0.01)
    s_w = st.slider("Sepal Width (cm)", 0.0, 10.0, 0.0, 0.01)
    p_l = st.slider("Petal Length (cm)", 0.0, 10.0, 0.0, 0.01)
    p_w = st.slider("Petal Width (cm)", 0.0, 10.0, 0.0, 0.01)

    # Your features must be in order that the model was trained on
    user_input = pd.DataFrame({
            'sepal_length': [s_l],
            'sepal_width': [s_w],
            'petal_length': [p_l],
            'petal_width': [p_w]
            })

    # Check out "pickling" to learn how we can "save" a model
    # and avoid the need to refit again!
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    X = df[features]
    y = df['species']

    # Model Selection
    model_option = st.selectbox("Select a Model", ["KNN", "Logistic Regression", "Random Forest"], index = None)

    if model_option:

        # Instantiating & fitting selected model
        if model_option == "KNN":
            k_value = st.slider("Select the number of neighbors (k)", 1, 21, 5, 2)
            model = KNeighborsClassifier(n_neighbors=k_value)
        elif model_option == "Logistic Regression":
            model = LogisticRegression()
        elif model_option == "Random Forest":
            model = RandomForestClassifier()
        
        if st.button("Make a Prediction!"):
            model.fit(X, y)
            prediction = model.predict(user_input)
            st.write(f"{model} predicts this iris flower is {prediction[0]} species!")
           # st.balloons()

            st.subheader("confusion matrix: ")
            ConfusionMatrixDisplay.from_estimator(model, X, y, cmap = "Blues")
            st.pyplot()
if page == "Extras":
    st.title("Adding Columns and Tabs")
  
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with col2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with col3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")

    st.markdown("---")
    st.divider()
    
    st.subheader("Adding tab")

    tab1, tab2, tab3 = st.tabs(["My TAB1", "My TAB2"," My TAB3"])
    with tab1:
        st.header("A cat")
        st.image("https://static.streamlit.io/examples/cat.jpg")

    with tab2:
        st.header("A dog")
        st.image("https://static.streamlit.io/examples/dog.jpg")

    with tab3:
        st.header("An owl")
        st.image("https://static.streamlit.io/examples/owl.jpg")

