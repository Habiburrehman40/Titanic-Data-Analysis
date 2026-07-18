# ============================================
# TITANIC DASHBOARD - Streamlit App
# Devixo Solutions - AI/ML Internship
# Name: Habib
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Titanic Dashboard - Habib",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    df = pd.read_csv(url)
    
    # Data Cleaning
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop('Cabin', axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns={'Sex': 'Gender'}, inplace=True)
    
    return df

df = load_data()

# ---------- SIDEBAR ----------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/800px-RMS_Titanic_3.jpg", use_column_width=True)
st.sidebar.title("🚢 Titanic Dashboard")
st.sidebar.markdown("---")

# Filters
st.sidebar.subheader("📊 Filters")
selected_pclass = st.sidebar.multiselect(
    "Passenger Class",
    options=sorted(df['Pclass'].unique()),
    default=sorted(df['Pclass'].unique())
)

selected_gender = st.sidebar.multiselect(
    "Gender",
    options=sorted(df['Gender'].unique()),
    default=sorted(df['Gender'].unique())
)

selected_survived = st.sidebar.multiselect(
    "Survival Status",
    options=sorted(df['Survived'].unique()),
    default=sorted(df['Survived'].unique())
)

age_range = st.sidebar.slider(
    "Age Range",
    min_value=int(df['Age'].min()),
    max_value=int(df['Age'].max()),
    value=(int(df['Age'].min()), int(df['Age'].max()))
)

fare_range = st.sidebar.slider(
    "Fare Range",
    min_value=0,
    max_value=int(df['Fare'].max()),
    value=(0, int(df['Fare'].max()))
)

# Apply filters
filtered_df = df[
    (df['Pclass'].isin(selected_pclass)) &
    (df['Gender'].isin(selected_gender)) &
    (df['Survived'].isin(selected_survived)) &
    (df['Age'] >= age_range[0]) &
    (df['Age'] <= age_range[1]) &
    (df['Fare'] >= fare_range[0]) &
    (df['Fare'] <= fare_range[1])
]

# ---------- MAIN CONTENT ----------
st.title("🚢 Titanic Data Analysis Dashboard")
st.markdown(f"*Showing {len(filtered_df)} out of {len(df)} total passengers*")
st.markdown("---")

# ---------- METRICS ROW ----------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Passengers",
        value=len(filtered_df),
        delta=f"{((len(filtered_df)/len(df))*100):.1f}% of total"
    )

with col2:
    survival_rate = (filtered_df['Survived'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    st.metric(
        label="Survival Rate",
        value=f"{survival_rate:.1f}%",
        delta=f"{survival_rate - 38.38:.1f}% vs overall"
    )

with col3:
    avg_age = filtered_df['Age'].mean()
    st.metric(
        label="Average Age",
        value=f"{avg_age:.1f} years",
        delta=f"{avg_age - df['Age'].mean():.1f} vs overall"
    )

with col4:
    avg_fare = filtered_df['Fare'].mean()
    st.metric(
        label="Average Fare",
        value=f"${avg_fare:.2f}",
        delta=f"${avg_fare - df['Fare'].mean():.2f} vs overall"
    )

st.markdown("---")

# ---------- CHARTS ROW 1 ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Survival Distribution")
    fig = px.pie(
        filtered_df,
        names='Survived',
        title='Survival Count',
        color='Survived',
        color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
        hole=0.3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("👥 Passenger Class Distribution")
    fig = px.bar(
        filtered_df['Pclass'].value_counts().reset_index(),
        x='Pclass',
        y='count',
        title='Passenger Class',
        color='Pclass',
        color_continuous_scale='Viridis',
        labels={'Pclass': 'Class', 'count': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHARTS ROW 2 ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📈 Age Distribution")
    fig = px.histogram(
        filtered_df,
        x='Age',
        nbins=20,
        title='Age Distribution',
        color_discrete_sequence=['#4dabf7']
    )
    fig.add_vline(x=filtered_df['Age'].mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {filtered_df['Age'].mean():.1f}")
    fig.add_vline(x=filtered_df['Age'].median(), line_dash="dash", line_color="green", annotation_text=f"Median: {filtered_df['Age'].median():.1f}")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("💰 Fare Distribution")
    fig = px.box(
        filtered_df,
        y='Fare',
        title='Fare Distribution',
        color='Pclass',
        color_discrete_sequence=['#ff6b6b', '#4dabf7', '#51cf66']
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------- CHARTS ROW 3 ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Survival by Gender")
    survival_gender = filtered_df.groupby('Gender')['Survived'].value_counts().unstack()
    fig = px.bar(
        survival_gender.reset_index(),
        x='Gender',
        y=[0, 1],
        title='Survival by Gender',
        labels={'value': 'Count', 'Gender': 'Gender'},
        color_discrete_map={'0': '#ff6b6b', '1': '#51cf66'}
    )
    fig.update_layout(legend_title_text='Survived')
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🎯 Survival by Class")
    survival_class = filtered_df.groupby('Pclass')['Survived'].value_counts().unstack()
    fig = px.bar(
        survival_class.reset_index(),
        x='Pclass',
        y=[0, 1],
        title='Survival by Passenger Class',
        labels={'value': 'Count', 'Pclass': 'Pclass'},
        color_discrete_map={'0': '#ff6b6b', '1': '#51cf66'}
    )
    fig.update_layout(legend_title_text='Survived')
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    st.plotly_chart(fig, use_container_width=True)

# ---------- HEATMAP ----------
st.subheader("🔗 Correlation Heatmap")
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
fig = px.imshow(
    filtered_df[numeric_cols].corr(),
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title='Correlation Matrix'
)
fig.update_layout(height=400)
st.plotly_chart(fig, use_container_width=True)

# ---------- SCATTER PLOT ----------
st.subheader("📌 Age vs Fare Scatter Plot")
fig = px.scatter(
    filtered_df,
    x='Age',
    y='Fare',
    color='Survived',
    size='Pclass',
    hover_data=['Gender'],
    title='Age vs Fare (Colored by Survival)',
    color_discrete_map={0: '#ff6b6b', 1: '#51cf66'},
    labels={'Survived': 'Survived'}
)
st.plotly_chart(fig, use_container_width=True)

# ---------- RAW DATA ----------
st.markdown("---")
st.subheader("📋 Filtered Data (Raw)")

if st.checkbox("Show Raw Data"):
    st.dataframe(
        filtered_df,
        column_config={
            "PassengerId": st.column_config.NumberColumn("ID"),
            "Survived": st.column_config.NumberColumn("Survived"),
            "Pclass": st.column_config.NumberColumn("Class"),
            "Gender": st.column_config.TextColumn("Gender"),
            "Age": st.column_config.NumberColumn("Age"),
            "SibSp": st.column_config.NumberColumn("Siblings"),
            "Parch": st.column_config.NumberColumn("Parents"),
            "Fare": st.column_config.NumberColumn("Fare"),
            "Embarked": st.column_config.TextColumn("Embarked")
        },
        use_container_width=True
    )
    st.caption(f"Showing {len(filtered_df)} rows out of {len(df)} total")

# ---------- DOWNLOAD BUTTON ----------
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data (CSV)",
        data=csv,
        file_name='titanic_filtered_data.csv',
        mime='text/csv',
        use_container_width=True
    )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Built with ❤️ by <b>Habib Ur Rehman</b> | Devixo Solutions - AI/ML Internship Task 01
    </div>
    """,
    unsafe_allow_html=True
)