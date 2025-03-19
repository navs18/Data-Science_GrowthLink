# Sales Prediction Task

## Task Objectives
This project aims to forecast product sales using machine learning based on historical sales data. The model is built using Python and the scikit-learn library.

## Dataset
The dataset used for this project is `car_purchasing.csv`, which contains information on factors like annual salary, credit card debt, and net worth to predict car purchase amounts.

## Steps to Run the Project

### Prerequisites
Ensured we have Python installed along with the required libraries. Install them using:
```sh
pip install -r requirements.txt
```

### Running the Model
1. Clonning the repository and navigating to the project folder.
2. Placing `car_purchasing.csv` in the directory.
3. Runnin the script:
```sh
python sales_prediction.py
```
4. The script will load data, preprocess it, train a RandomForest model, and output evaluation metrics along with a visualization of actual vs predicted sales.

## Model & Approach
- **Preprocessing**: Handling missing values, feature selection, and scaling.
- **Model Used**: RandomForestRegressor
- **Evaluation Metrics**: MAE, MSE, R2 Score

## Expected Outcome
A model that helps businesses optimize marketing strategies for sales growth by providing accurate sales predictions.

## Tools & Libraries Used

Pandas: https://pandas.pydata.org/

NumPy: https://numpy.org/

Matplotlib: https://matplotlib.org/

Seaborn: https://seaborn.pydata.org/

Scikit-learn: https://scikit-learn.org/

