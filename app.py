from flask import Flask, render_template, request
from flask import Flask, render_template, request, redirect, url_for, flash
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models and scalers
with open('cropmodel.pkl', 'rb') as model_file:
    crop_model = pickle.load(model_file)

with open('cropminmaxscaler.pkl', 'rb') as scaler_file:
    crop_scaler = pickle.load(scaler_file)

with open('crop_yield_model.pkl', 'rb') as model_file:
    crop_yield_model = pickle.load(model_file)

with open('crop_yield_scaler.pkl', 'rb') as scaler_file:
    crop_yield_scaler = pickle.load(scaler_file)

with open('Region_label_encoder.pkl', 'rb') as f:
    region_encoder = pickle.load(f)

with open('Soil_Type_label_encoder.pkl', 'rb') as f:
    soil_type_encoder = pickle.load(f)

with open('Crop_label_encoder.pkl', 'rb') as f:
    Crop_label_encoder = pickle.load(f)

with open('Weather_Condition_label_encoder.pkl', 'rb') as f:
    Weather_Condition_label_encoder = pickle.load(f)

with open('crop_market_model.pkl', 'rb') as model_file:
    crop_market_model = pickle.load(model_file)

with open('crop_market_minmax1_scaler.pkl', 'rb') as model_file:
    crop_market_minmax_scaler = pickle.load(model_file)

with open('state_label_encoder.pkl', 'rb') as f:
    state_encoder = pickle.load(f)

with open('district_label_encoder.pkl', 'rb') as f:
    district_encoder = pickle.load(f)

with open('market_label_encoder.pkl', 'rb') as f:
    market_encoder = pickle.load(f)

with open('commodity_label_encoder.pkl', 'rb') as f:
    commodity_encoder = pickle.load(f)

with open('variety_label_encoder.pkl', 'rb') as f:
    variety_encoder = pickle.load(f)

@app.route('/')
def home():
      return render_template('index.html')

# @app.route('/register', methods=['GET', 'POST'])
# def register():
#     if request.method == 'POST':
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']
        
#         # Simple validation (replace with your own logic)
#         if password != confirm_password:
#             flash('Passwords do not match!', 'danger')
#             return redirect(url_for('register'))
        
#         # Add user registration logic here (e.g., save to database)
#         flash('Registration successful! Please log in.', 'success')
#         return redirect(url_for('login'))
    
#     return render_template('register.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
        
#         # Example login check (replace with actual authentication)
#         if username == 'admin' and password == 'password':
#             flash('Login successful!', 'success')
#             return redirect(url_for('home'))
#         else:
#             flash('Invalid credentials. Please try again.', 'danger')
#             return redirect(url_for('login'))

#     return render_template('login.html')

# app.route('/home')
# def dashboard():
#     # Main dashboard page after login
#     return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Crop mapping dictionaries
crop_dict = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
    'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
    'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
    'apple': 16, 'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

# Define the recommendation function
def recommendation(N, P, K, humidity, ph, temperature_category, rainfall_category):
    # Convert categories back to numeric values
    temperature_dict = {
        'Cool': 10,
        'Mild': 15,
        'Warm': 20,
        'Hot': 25,
        'Very Hot': 30,
        'Extreme Heat': 35
    }
    
    rainfall_dict = {
        'No Rain': 0,
        'Light Rain': 1,
        'Moderate Rain': 5,
        'Heavy Rain': 10,
        'Very Heavy Rain': 20,
        'Extreme Rain': 30
    }
    
    # Get the numeric values for temperature and rainfall
    temperature = temperature_dict.get(temperature_category, 0)  # Default to 0 if not found
    rainfall = rainfall_dict.get(rainfall_category, 0)  # Default to 0 if not found

    features = np.array([[N, P, K, humidity, ph, temperature, rainfall]])
    scaled_features = crop_scaler.transform(features)  # Use the scaler to transform
    prediction = crop_model.predict(scaled_features).reshape(1, -1)
    label_name = reverse_crop_dict[prediction[0][0]]  # Convert numeric back to label
    return label_name

@app.route('/crop_recommendation', methods=['GET', 'POST'])
def index():
    crop_result = None
    if request.method == 'POST':
        # Crop Recommendation Form Submission
        if 'N' in request.form:
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            humidity = float(request.form['humidity'])
            ph = float(request.form['ph'])
            temperature_category = request.form['temperature_category']
            rainfall_category = request.form['rainfall_category']
            crop_result = recommendation(N, P, K, humidity, ph, temperature_category, rainfall_category)
    
    return render_template('index.html', crop_result=crop_result, model="Crop Recommendation")

@app.route('/yield_prediction', methods=['POST'])
def yield_prediction():
    yield_result = None
    if request.method == 'POST':
        # Retrieve categorical inputs from the form
        region = request.form['Region']
        soil_type = request.form['Soil_Type']
        crop = request.form['Crop']
        weather_condition = request.form['Weather_Condition']
        
        # Use label encoders to convert strings to numeric values
        try:
            region_encoded = region_encoder.transform([region])[0]
            soil_type_encoded = soil_type_encoder.transform([soil_type])[0]
            crop_encoded = Crop_label_encoder.transform([crop])[0]
            weather_condition_encoded = Weather_Condition_label_encoder.transform([weather_condition])[0]
        except ValueError as e:
            # Handle case where encoding fails due to an unknown label
            return render_template('index.html', result=f"Encoding error: {e}", model="Yield Prediction")

        # Retrieve and process numeric inputs
        try:
            Rainfall_mm = float(request.form['Rainfall_mm'])
            Temperature_Celsius = float(request.form['Temperature_Celsius'])
            fertilizer_used = 1 if request.form['Fertilizer_Used'] == 'True' else 0
            irrigation_used = 1 if request.form['Irrigation_Used'] == 'True' else 0
            days_to_harvest = int(request.form['Days_to_Harvest'])
        except ValueError as e:
            # Handle case where conversion to float/int fails
            return render_template('index.html', result=f"Input error: {yield_result}", model="Yield Prediction")

        # Assemble all encoded and numeric features
        features = np.array([[region_encoded, soil_type_encoded, crop_encoded, Rainfall_mm,
                              Temperature_Celsius, fertilizer_used, irrigation_used,
                              weather_condition_encoded, days_to_harvest]])

        # Ensure all features are numeric before scaling
        try:
            scaled_features = crop_yield_scaler.transform(features)
        except ValueError as e:
            # Handle errors during scaling
            return render_template('index.html', result=f"Scaling error: {e}", model="Yield Prediction")
        
        # Predict yield
        yield_result = crop_yield_model.predict(scaled_features)[0]
        
        return render_template('index.html', yield_result=f"{yield_result:.2f} tons/ha", model="Yield Prediction")

@app.route('/market_price_prediction', methods=['POST'])
def market_price_prediction():
    price_result = None
    if request.method == 'POST':
        # Retrieve categorical form data
        state = request.form.get('state')
        district = request.form.get('district')
        market = request.form.get('market')
        commodity = request.form.get('commodity')
        variety = request.form.get('variety')

        # Check if all required fields are provided
        if not all([state, district, market, commodity, variety]):
            return render_template('index.html', result="Error: Missing form data", model="Market Price Prediction")

        # Retrieve and parse numerical inputs
        try:
            min_price = float(request.form.get('min_price', 0))
            max_price = float(request.form.get('max_price', 0))
            arrival_day = int(request.form.get('arrival_day', 1))
            arrival_month = int(request.form.get('arrival_month', 1))
            arrival_year = int(request.form.get('arrival_year', 2000))
        except (ValueError, TypeError):
            return render_template('index.html', result="Error: Invalid numerical input", model="Market Price Prediction")

        # Encode categorical features
        try:
            state_encoded = state_encoder.transform([state])[0]
            district_encoded = district_encoder.transform([district])[0]
            market_encoded = market_encoder.transform([market])[0]
            commodity_encoded = commodity_encoder.transform([commodity])[0]
            variety_encoded = variety_encoder.transform([variety])[0]
        except ValueError as e:
            return render_template('index.html', result=f"Encoding error: {e}", model="Market Price Prediction")

        # Prepare the features for prediction
        features = np.array([[state_encoded, district_encoded, market_encoded, commodity_encoded, variety_encoded,
                              min_price, max_price, arrival_day, arrival_month, arrival_year]])

        # Only scale the relevant numerical features (e.g., min_price and max_price)
        numerical_features = features[:, 5:7]  # Extract the 'min_price' and 'max_price' columns (index 5 and 6)
        
        # Ensure we are passing only 2 features to the scaler
        if numerical_features.shape[1] != 2:
            return render_template('index.html', result="Error: Incorrect number of features for scaling", model="Market Price Prediction")
        
        # Scale the numerical features (min_price and max_price)
        scaled_numerical_features = crop_market_minmax_scaler.transform(numerical_features)

        # Replace the original numerical columns with the scaled ones
        features[:, 5:7] = scaled_numerical_features

        # Ensure the features have the correct number of elements (10 features)
        if features.shape[1] != 10:
            return render_template('index.html', result="Error: Incorrect number of features for prediction", model="Market Price Prediction")

        # Predict market price using the model
        price_result = crop_market_model.predict(features)[0]

        return render_template('index.html', price_result=price_result, model="Market Price Prediction")
    
if __name__ == "__main__":
    app.run(debug=True)
