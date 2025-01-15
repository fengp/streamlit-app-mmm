import streamlit as st
import pandas as pd 

st.title("😎 Check this out! My new web app to facilitate Marketing Mix Modeling analysis!")
st.write(
    "The goal of this project is to leverage marketing mix modeling analysis to help marketers and advertisers to optimize their budget allocations among diverse media channels to achieve the maximium return on ad spend (ROAS)."
)
st.write("Thanks to Stream Cloud Community that inspired me for this project [docs.streamlit.io](https://docs.streamlit.io/).")
st.write ("Also, special thanks to Pymc-marketing open source library")

st. title ("MMM Advertising Analytics - Basic")
uploaded_file = st.file_uploader ("Upload your advertising csv file", type = "csv")
if uploaded_file is not None: 
    dat = pd.read_csv(uploaded_file)
    st.write ("Uploaded Data:")
    st.dataframe(dat)


#speed up the loading data by cache the data
#pretend the file is loaded and display the data in the csv
@st.cache_data
def load_data ():
    return pd.read_csv ("Advertising.csv")

df = load_data()

st.header("Load a sample data to inspect the data ")
st.write ("I selected a sample data from Kaggle to display here for Demo purpose")
st.write ("'st.data_editor' allows us to display and edit the data")
st.data_editor(df)


#check the shape of the data
st.write("Checking dataset shape...")
st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

# Check for missing values
st.write("Checking for missing values...")

# Calculate missing values
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

# Display the missing values summary
st.write("Missing Values (Count):")
st.write(missing_values[missing_values > 0])

st.write("Missing Values (%):")
st.write(missing_percentage[missing_percentage > 0])

# Visualize missing values
if missing_values.sum() > 0:
    st.write("Missing Values Visualization:")
    fig, ax = plt.subplots()
    missing_values[missing_values > 0].plot(kind='bar', ax=ax)
    ax.set_title("Missing Values per Column")
    ax.set_ylabel("Count of Missing Values")
    ax.set_xlabel("Columns")
    st.pyplot(fig)
else:
     st.success("No missing values found in the dataset!")



import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.api as sm 


# Function to calculate regression metrics
def regression_metrics(X, y):
 X_with_const = sm.add_constant(X)
 model = sm.OLS(y, X_with_const).fit()
 return model


#Function to create plot
def plot_residuals(y, predictions):
 residuals = y - predictions
 plt.figure(figsize=(10, 6))
 sns.residplot(x=predictions, y=residuals, lowess=True,
              scatter_kws={'color': 'gray', 'alpha': 0.7}, 
              line_kws={'color': 'red', 'lw': 2})
 plt.xlabel('Predicted Values')
 plt.ylabel('Residuals')
 plt.title('Residual Plot')
 st.pyplot(plt.gcf())


 # Function to plot normal probability plot
def plot_normal_prob(residuals):
 sm.qqplot(residuals, line='45')
 plt.title('Normal Probability Plot')
 st.pyplot(plt.gcf())

 # Function to plot predicted vs observed values
def plot_predicted_vs_observed(y, predictions):
 plt.figure(figsize=(10, 6))
 plt.scatter(y, predictions, alpha=0.7)
 plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
 plt.xlabel('Observed Values')
 plt.ylabel('Predicted Values')
 plt.title('Predicted vs Observed Plot')
 st.pyplot(plt.gcf())

 # Streamlit app
st.title("Multivariate Regression Analysis")

 # Calculate total sum for each column
st.subheader("Total sum of each column:")
totals = df.drop(columns=['week']).sum()
st.write(totals)

# Prepare data for regression analysis 
X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

#Initial regression analysis 
st.subheader ('Initial regerssion Analysis')
model = regression_metrics (X,y)
st.write (model.summary())

# Final model metrics
st.subheader("Final Model Metrics")
st.write(f"R-squared: {model.rsquared}")
st.write(f"F-statistic: {model.fvalue}")
st.write(f"Intercept: {model.params['const']}")
for variable, coef in model.params.items():
  if variable != 'const':
    st.write(f"Slope for {variable}: {coef}")

st.write ("Note: all 3 media channels are statistically significant, however, if we look at the coefficient, newspaper has negative correlations while both TV and radio are positively correlated to sales growth ")

p_value = model.pvalues[variable]
st.write(f"P-value for {variable}: {p_value}")
if p_value < 0.05:
  st.write(f"{variable} is statistically significant at 95% confidence interval")


# Predictions and residuals
predictions = model.predict(sm.add_constant(X))
residuals = y - predictions 

st.subheader("Residual Plot")
plot_residuals(y,predictions)

#Plot normal probability plot
st.subheader ("Normal Probability Plot")
plot_normal_prob (residuals)

#Plot predicted vs. observed values
st.subheader ("Predicted vs.Observed Plot")
plot_predicted_vs_observed(y,predictions)
st. write ("However, this is just a simiple mmm analysis, and more transformation needs to be done to cover ad stock and statuaration and more realistic datasets and confounding elements such as seasonality and product price need to be considered.")