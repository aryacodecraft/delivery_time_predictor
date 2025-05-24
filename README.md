# 🚚 Delivery Time Prediction Model

A machine learning model that predicts food delivery times based on various factors like distance, weather, traffic conditions, and more.

## 🎯 Features

- Predicts delivery time in minutes
- Uses Random Forest Regressor
- Handles multiple input features:
  - Distance (km)
  - Weather conditions
  - Traffic level
  - Time of day
  - Vehicle type
  - Preparation time
  - Courier experience

## 📊 Model Performance

- Mean Absolute Error (MAE): 7.05 minutes
- Root Mean Squared Error (RMSE): 9.87
- R² Score: 0.77

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/delivery_predictor.git
cd delivery_predictor
```

2. Create and activate virtual environment (optional but recommended):
```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For Linux/Mac
source venv/bin/activate
```

3. Install required packages:
```bash
pip install pandas numpy scikit-learn jupyter
```

## 💻 Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook model_training.ipynb
```

2. Run all cells in the notebook to:
   - Load and preprocess the data
   - Train the model
   - Save the trained model and label encoders

3. For predictions, use this sample code:
```python
import pandas as pd
import pickle

# Load the saved model and label encoders
model = pickle.load(open('model.pkl', 'rb'))
labels = pickle.load(open('label_encoders.pkl', 'rb'))

# Prepare your input
sample_input = [[7.8, 'Rainy', 'High', 'Evening', 'Bike', 20, 3]]
features = ['Distance_km','Weather', 'Traffic_Level', 'Time_of_Day', 
            'Vehicle_Type', 'Preparation_Time_min', 'Courier_Experience_yrs']

# Create DataFrame and transform categorical variables
input_df = pd.DataFrame(sample_input, columns=features)
for col in ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']:
    input_df[col] = labels[col].transform(input_df[col])

# Get prediction
prediction = model.predict(input_df)
print(f"Predicted Delivery Time: {prediction[0]} minutes")
```

## 📝 Dataset Description

The model is trained on a dataset containing the following features:
- Distance_km: Distance in kilometers
- Weather: Weather conditions (Clear, Rainy, Windy, Foggy, Snowy)
- Traffic_Level: Traffic conditions (Low, Medium, High)
- Time_of_Day: Time period (Morning, Afternoon, Evening, Night)
- Vehicle_Type: Delivery vehicle (Bike, Scooter)
- Preparation_Time_min: Food preparation time in minutes
- Courier_Experience_yrs: Courier's experience in years

## 📁 Files
- `model_training.ipynb`: Jupyter notebook with model training code
- `model.pkl`: Trained Random Forest model
- `label_encoders.pkl`: Label encoders for categorical variables
- `Food_Delivery_Times.csv`: Training dataset

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.