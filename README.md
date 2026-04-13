Employee Salary Prediction Web App
This project is a machine learning application that predicts employee salaries based on various professional factors. The project includes a data analysis and model training pipeline in a Jupyter Notebook and a user-friendly web application built with Streamlit for interactive predictions and data exploration.

🚀 Features
Salary Prediction: Predicts employee salaries in USD based on experience level, employment type, job title, company location, and company size.

Currency Conversion: Converts the predicted USD salary to other currencies using real-time exchange rates.

Interactive Visualizations: Explore salary data through various charts and plots.

User-Friendly Interface: A clean and intuitive web interface for easy interaction.

📊 Dataset
The project uses the salary.csv dataset, which contains information about salaries for various tech roles. The key features used for prediction are:

experience_level

employment_type

job_title

company_location

company_size

🛠️ Methodology
Data Cleaning: The initial dataset was cleaned to handle missing values and format inconsistencies.

Feature Engineering: Categorical features were encoded into numerical values suitable for machine learning models.

Model Training: Several regression models were trained and evaluated, including:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

Decision Tree Regressor

Support Vector Regressor (SVR)

Model Selection: The Gradient Boosting Regressor was selected as the best-performing model based on the Mean Absolute Error (MAE).

Deployment: The trained model was saved and integrated into a Streamlit web application for easy use.

⚙️ Installation
To run this project locally, follow these steps:

Clone the repository:

git clone https://github.com/Xema7/Employee-Salary-Prediction-Ravi-Asopa.git

Navigate to the project directory:

cd employee-salary-prediction

Install the required dependencies:

pip install -r requirements.txt

▶️ Usage
To run the Streamlit web application, use the following command in your terminal:

streamlit run app.py

💻 Technologies Used
Python: The core programming language.

Pandas: For data manipulation and analysis.

Scikit-learn: For machine learning model training and evaluation.

Streamlit: To create the interactive web application.

Matplotlib & Seaborn: For data visualization.

Jupyter Notebook: For data exploration and model development.
