import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.title("Mall Customer Segmentation")

# Upload dataset
uploaded_file = st.file_uploader("D:\project\Mall_Customers.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.dataframe(df.head())

    X = df[['Age','Annual Income (k$)','Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    k = st.slider("Select number of clusters",2,10,5)

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)

    fig, ax = plt.subplots()

    ax.scatter(df['Annual Income (k$)'],
               df['Spending Score (1-100)'],
               c=df['Cluster'],
               cmap='viridis')

    ax.set_xlabel("Annual Income")
    ax.set_ylabel("Spending Score")

    st.pyplot(fig)

    st.write("Cluster Summary")
    st.write(df.groupby("Cluster").mean(numeric_only=True))