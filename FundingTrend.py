import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
import pickle

st.set_page_config(layout="wide")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("./TABLE1.csv")
    gdp = pd.read_csv("./gdp_monthly.csv")
    table2 = pd.read_csv("./TABLE2.csv")  # New data for second page
    data['YearMonth'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Publish_month'].astype(int).astype(str), errors='coerce', format='%Y-%m')
    return data, gdp, table2

data, gdp, table2 = load_data()




# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [ "Citation Analysis" , "Funded Articles Prediction" , "Geospatial Dashboard" , "Has Funding Prediction"])



if page == "Funded Articles Prediction":
    # --- Page 1: Funded Articles Prediction ---
    st.title("Funded Articles Prediction by Subject Area")
    
    # Subject areas list
    subject_areas = ['Agricultural and Biological Sciences',
                     'Arts and Humanities',
                     'Biochemistry, Genetics and Molecular Biology',
                     'Business, Management and Accounting',
                     'Chemical Engineering',
                     'Chemistry',
                     'Computer Science',
                     'Decision Sciences',
                     'Dentistry',
                     'Earth and Planetary Sciences',
                     'Economics, Econometrics and Finance',
                     'Energy',
                     'Engineering',
                     'Environmental Science',
                     'Health Professions',
                     'Immunology and Microbiology',
                     'Materials Science',
                     'Mathematics',
                     'Medicine',
                     'Multidisciplinary',
                     'Neuroscience',
                     'Nursing',
                     'Pharmacology, Toxicology and Pharmaceutics',
                     'Physics and Astronomy',
                     'Psychology',
                     'Social Sciences',
                     'Veterinary']

    selected_area = st.selectbox("Select a subject area:", subject_areas)

    # Filter data by selected subject area
    subject_country_data = data.groupby(['YearMonth', 'Subject_Field'])['Eid'].count().reset_index()
    subject_country_data = subject_country_data[subject_country_data['Subject_Field'] == selected_area]

    # Merge with GDP data
    gdp['YearMonth'] = pd.to_datetime(gdp['YearMonth'])
    subject_country_data = subject_country_data.merge(gdp, on="YearMonth", how="left")

    # Set index and handle outliers
    subject_country_data.set_index('YearMonth', inplace=True)
    subject_country_data = subject_country_data.asfreq('MS')
    Q1 = subject_country_data['Eid'].quantile(0.1)
    Q3 = subject_country_data['Eid'].quantile(0.9)
    subject_country_data['Eid'] = subject_country_data['Eid'].apply(lambda x: Q1 if x < Q1 else (Q3 if x > Q3 else x))

    # Prepare data for Prophet
    df = subject_country_data.reset_index()
    df_prophet = df.rename(columns={'YearMonth': 'ds', 'Eid': 'y'})
    train = df_prophet.iloc[:-4]

    # Prophet model
    model = Prophet(changepoint_prior_scale=0.5, seasonality_prior_scale=1.0, n_changepoints=10)
    model.add_regressor('gdp')
    model.fit(train)

    # Future dataframe
    future = model.make_future_dataframe(periods=11, freq='MS')
    future = future.merge(gdp, how='left', left_on='ds', right_on='YearMonth').drop(columns="YearMonth")

    # Forecast
    forecast = model.predict(future)

    # Display results
    st.subheader(f"Trend for {selected_area}")
    fig1 = model.plot(forecast)
    fig1.suptitle(f"{selected_area} Funded Articles", fontsize=16, y=0.9)
    ax = fig1.gca()
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Articles", fontsize=12)
    st.pyplot(fig1)

elif page == "Citation Analysis":
    # --- Page 2: Citation Analysis ---
    st.title("Citation Analysis by Funding Status")

    # Average Citation Count
    mean_citations = table2.groupby('Has_Funding')['Cite-By_Count'].mean()
    categories = ['Not Funded', 'Funded']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, mean_citations, color=['skyblue', 'lightgreen'])

    # Total Citation Count
    total_citations = table2.groupby('Has_Funding')['Cite-By_Count'].sum()

    plt.figure(figsize=(8, 6))
    bars = plt.bar(categories, total_citations, color=['coral', 'lightblue'])

    # Add value annotations
    for bar, value in zip(bars, total_citations):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 9000, 
                 f'{int(value)}', ha='center', va='bottom', fontsize=12, color='black')

    plt.title("Total Citation Count by Funding Status", fontsize=14)
    plt.ylabel("Total Citation Count", fontsize=12)
    plt.xlabel("Funding Status", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

elif page == "Geospatial Dashboard":
    st.title("Geospatial analysis")
    st.write("Powered by power bi")

    # Replace the URL below with your actual Power BI Embed link
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=7af25db8-1897-408e-ad66-43b4b6a181e2&autoAuth=true&ctid=271d5e7b-1350-4b96-ab84-52dbda4cf40c"

    # Embed Power BI using an iframe
    st.components.v1.html(
        f"""
        <iframe src="{power_bi_url}" width="100%" height="1000px" frameborder="0" allowfullscreen="true"></iframe>
        """,
        height=1000,
    )
elif page == "Has Funding Prediction":
    # --- Page 4: Has Funding Prediction ---
    st.title("Has Funding Prediction")

    # Load the model and transformers
    with open("rf_model.pkl", "rb") as f:
        rf_model = pickle.load(f)
    with open("text_transformer.pkl", "rb") as f:
        text_transformer = pickle.load(f)
    with open("mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    # User input
    title = st.text_input("Enter the Title of the Research")
    index_terms = st.text_area("Enter Index Terms (comma-separated)")
    subject_field = st.text_area("Enter Subject Fields (comma-separated)")

    if st.button("Predict"):
        # Process user input
        text_all = title + " " + index_terms
        subject_field_list = [field.strip() for field in subject_field.split(',')]

        # Transform input features
        text_tfidf = text_transformer.transform([text_all])
        subject_binarized = mlb.transform([subject_field_list])

        # Combine features
        combined_features = hstack([text_tfidf, subject_binarized])

        # Predict
        prediction = rf_model.predict(combined_features)[0]

        # Display result
        result = "Has Funding" if prediction == 1 else "No Funding"
        st.write(f"Prediction: {result}")
    example_data = pd.DataFrame({
        'Title': ["Lower Serum Zinc and Higher CRP Strongly Predict Prenatal Depression and Physio-somatic Symptoms, Which All Together Predict Postnatal Depressive Symptoms"],
        'Index_Terms': ["Adult Article Beck Depression Inventory Biomarkers C reactive protein C-Reactive Protein Depression, Postpartum Edinburgh Postnatal Depression Scale Fatigue Female Hamilton Depression Rating Scale Humans Inflammation Mediators Medically Unexplained Symptoms Predictive Value of Tests Pregnancy Pregnancy Complications Prenatal Care Psychiatric Status Rating Scales State Trait Anxiety Inventory Zinc adult anxiety disorder autacoid backache biological marker blood body mass constipation controlled study disease severity dyspepsia dysthymia fatigue female first trimester pregnancy haptoglobin human immune activation immune inflammatory pathophysiology immune response major clinical study major depression medically unexplained symptom metabolism myalgia pathophysiology perinatal depression postnatal depression prediction predictive value pregnancy pregnancy complication pregnant woman premenstrual syndrome prenatal care procedures protein blood level psychological rating scale puerperal depression second trimester pregnancy symptom zinc zinc blood level"],
        'Subject_Field': ["Neuroscienceg"]
    })

    # Preprocess ข้อมูลตัวอย่าง
    example_data['Index_Terms'] = example_data['Index_Terms'].fillna("").astype(str)
    example_data['text_all'] = example_data['Title'].fillna("") + " " + example_data['Index_Terms']
    example_data['Subject_Field'] = example_data['Subject_Field'].apply(lambda x: [field.strip() for field in x.split(',')])

    # Transform text features (assuming text_transformer และ mlb ถูกเทรนไว้แล้ว)
    example_text_tfidf = text_transformer.transform(example_data['text_all'])
    example_subject_binarized = mlb.transform(example_data['Subject_Field'])

    # Combine features
    example_combined_features = hstack([example_text_tfidf, example_subject_binarized])

    # ทำนายผลลัพธ์
    example_predictions = rf_model.predict(example_combined_features)

    # แสดงผลลัพธ์
    example_data['Prediction'] = ["Has Funding" if pred == 1 else "No Funding" for pred in example_predictions]
    print(example_data[['Title', 'Prediction']])