import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import matplotlib.pyplot as plt

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("./TABLE1.csv")
    gdp = pd.read_csv("./gdp_monthly.csv")
    data['YearMonth'] = pd.to_datetime(data['Year'].astype(str) + '-' + data['Publish_month'].astype(int).astype(str), errors='coerce', format='%Y-%m')
    return data, gdp

data, gdp = load_data()

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


# Streamlit UI
st.title("Funded Articles Prediction by Subject Area")
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

# # Cross-validation metrics
# st.subheader("Model Performance")
# df_cv = cross_validation(model, initial='1920 days', period='180 days', horizon='365 days')
# df_p = performance_metrics(df_cv)
# st.dataframe(df_p)

# # MAPE Plot
# fig2 = plot_cross_validation_metric(df_cv, metric='mape')
# st.pyplot(fig2)
