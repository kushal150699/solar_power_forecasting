import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from datetime import datetime, timedelta
import plotly.graph_objects as go
import requests
import pytz
from pvlib import solarposition
import pvlib
from sklearn.preprocessing import  OrdinalEncoder, StandardScaler

# Load the model and scalers
@st.cache_resource
def load_model():
    model = joblib.load('xgb.pkl')
    return model

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather_forecast(latitude, longitude, start_date, end_date,timezone="Australia/Darwin"):
    """Fetch weather forecast data from Open-Meteo API, including one day prior to the start date"""
    base_url = "https://api.open-meteo.com/v1/forecast"
    
    # Calculate one day prior to the start_date
    previous_day = start_date - timedelta(days=2)
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation"
        ],
        "timezone": timezone,
        "start_date": previous_day.strftime("%Y-%m-%d"),  # Use previous day as the start date
        "end_date": end_date.strftime("%Y-%m-%d")
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_valid_date_range():
    """Get valid date range for forecasting"""
    today = datetime.now().date()
    max_date = today + timedelta(days=14)
    return today, max_date

def process_weather_data(weather_data):
    """Process weather data into a pandas DataFrame"""
    if not weather_data:
        return None
    
    df = pd.DataFrame()
    hourly = weather_data['hourly']
    
    df['date'] = pd.to_datetime(hourly['time'])
    df['temperature_2m'] = hourly['temperature_2m']
    df['relativehumidity_2m'] = hourly['relative_humidity_2m']
    df['dew_point_2m'] = hourly['dew_point_2m']
    df['surface_pressure'] = hourly['surface_pressure']
    df['cloud_cover'] = hourly['cloud_cover']
    df['wind_speed_10m'] = hourly['wind_speed_10m']
    df['wind_direction_10m'] = hourly['wind_direction_10m']
    df['shortwave_radiation'] = hourly['shortwave_radiation']
    df['direct_radiation'] = hourly['direct_radiation']
    df['diffuse_radiation'] = hourly['diffuse_radiation']
    
    return df

def add_engineered_features(df, start_date,latitude, longitude , altitude):

    """Add engineered features to match model expectations exactly"""
    # Calculate temperature features
    df['temperature_f'] = df['temperature_2m'] * 9/5 + 32
    
    # Calculate THI (Temperature-Humidity Index)
    df['THI'] = df['temperature_f'] - (0.55 - 0.0055 * df['relativehumidity_2m']) * (df['temperature_f'] - 58)
    
    # Calculate wind features
    df['wind_speed_mph'] = df['wind_speed_10m'] * 2.237
    df['wind_chill'] = (35.74 + 0.6215*df['temperature_f'] - 
                       35.75*(df['wind_speed_mph']**0.16) + 
                       0.4275*df['temperature_f']*(df['wind_speed_mph']**0.16))
    
    # Calculate heat index using simplified formula
    df['heat_index'] = -42.379 + 2.04901523*df['temperature_f'] + 10.14333127*df['relativehumidity_2m']
    
    # Add solar position parameters
    times = pd.DatetimeIndex(df['date'])
    site = pvlib.location.Location(latitude, longitude, altitude=altitude)
    solar_position = site.get_solarposition(times)
    solar_zenith_angles = solar_position['zenith']
    df['solar_zenith_angle'] = solar_zenith_angles.values
    
    df['air_mass'] = 1 / np.cos(np.radians(df['solar_zenith_angle']))
    
    # Add time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['hour_of_day'] = df['date'].dt.hour
    df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    
    # Add season (Southern Hemisphere)
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # Summer
        elif month in [3, 4, 5]:
            return 1  # Autumn
        elif month in [6, 7, 8]:
            return 2  # Winter
        else:
            return 3  # Spring
    df['season'] = df['date'].dt.month.apply(get_season)
    ord_enc = joblib.load('season.pkl')
    season = ord_enc.fit_transform(np.array(df['season']).reshape(-1,1))
    df['season'] = season

    # Add time intervals
    def detect_time_interval(df):
        intervals = {'first_interval': (6, 9), 'second_interval': (9, 11), 'third_interval': (11, 13),
                    'fourth_interval': (13, 15), 'fifth_interval': (15, 17), 'sixth_interval': (17, 20)}
        df['time_interval'] = pd.cut(df['hour_of_day'], bins=[interval[0] for interval in intervals.values()] + [24],
                                    labels=[interval_name for interval_name in intervals.keys()],
                                    include_lowest=True, right=False)
        return df
    
    df = detect_time_interval(df)
    
    # Add lag features
    for col in ['temperature_2m', 'cloud_cover', 'wind_speed_10m']:
        df[f'{col}_lag_1h'] = df[col].shift(1)
        df[f'{col}_lag_24h'] = df[col].shift(24)
        df[f'{col}_rolling_mean_24h'] = df[col].rolling(window=24, min_periods=1).mean()
        df[f'{col}_rolling_std_24h'] = df[col].rolling(window=24, min_periods=1).std()
        df[f'{col}_change'] = df[col].diff()
    
    # Add interaction terms
    df['temp_wind_interaction'] = df['temperature_2m'] * df['wind_speed_10m']
    df['cloud_radiation_interaction'] = df['cloud_cover'] * df['direct_radiation']
    
    # Calculate weather stability index
    df['weather_stability_index'] = (df['temperature_2m_change'].abs() + 
                                   df['cloud_cover_change'].abs() + 
                                   df['wind_speed_10m_change'].abs())
    
    # Add placeholder values for power-related features
    df['Active_Power_lag_1h'] = df['direct_radiation'] * 0.1  # Approximate based on radiation
    df['Active_Power_lag_24h'] = df['direct_radiation'] * 0.1  # Approximate based on radiation
    
    # Select and order features exactly as specified
    feature_columns = [
        'temperature_2m', 'relativehumidity_2m', 'dew_point_2m',
        'surface_pressure', 'cloud_cover', 'wind_speed_10m',
        'wind_direction_10m', 'shortwave_radiation', 'direct_radiation',
        'diffuse_radiation', 'season', 'temperature_f', 'THI', 'wind_speed_mph',
        'wind_chill', 'heat_index', 'solar_zenith_angle', 'air_mass',
        'day_of_year', 'day_of_year_sin', 'day_of_year_cos', 'hour_of_day',
        'hour_of_day_sin', 'hour_of_day_cos', 'Active_Power_lag_1h',
        'Active_Power_lag_24h', 'temperature_2m_lag_1h',
        'temperature_2m_lag_24h', 'cloud_cover_lag_1h', 'cloud_cover_lag_24h',
        'wind_speed_10m_lag_1h', 'wind_speed_10m_lag_24h',
        'temperature_2m_rolling_mean_24h', 'temperature_2m_rolling_std_24h',
        'cloud_cover_rolling_mean_24h', 'cloud_cover_rolling_std_24h',
        'wind_speed_10m_rolling_mean_24h', 'wind_speed_10m_rolling_std_24h',
        'temp_wind_interaction', 'cloud_radiation_interaction',
        'temperature_2m_change', 'cloud_cover_change', 'wind_speed_10m_change',
        'weather_stability_index', 'time_interval'
    ]
    df = df.dropna()
    df = df[df['date'] >= pd.Timestamp(start_date)]
    df.reset_index(drop=True, inplace=True)
    return df[feature_columns]

def standardize_data(new_test):

    standardize_predictor_list = ['temperature_2m', 'relativehumidity_2m', 'dew_point_2m',
       'surface_pressure', 'cloud_cover', 'wind_speed_10m',
       'wind_direction_10m', 'shortwave_radiation', 'direct_radiation',
       'diffuse_radiation', 'season', 'temperature_f', 'THI',
       'wind_speed_mph', 'wind_chill', 'heat_index', 'solar_zenith_angle',
       'air_mass', 'day_of_year', 'day_of_year_sin', 'day_of_year_cos',
       'hour_of_day', 'hour_of_day_sin', 'hour_of_day_cos',
       'Active_Power_lag_1h', 'Active_Power_lag_24h', 'temperature_2m_lag_1h',
       'temperature_2m_lag_24h', 'cloud_cover_lag_1h', 'cloud_cover_lag_24h',
       'wind_speed_10m_lag_1h', 'wind_speed_10m_lag_24h',
       'temperature_2m_rolling_mean_24h', 'temperature_2m_rolling_std_24h',
       'cloud_cover_rolling_mean_24h', 'cloud_cover_rolling_std_24h',
       'wind_speed_10m_rolling_mean_24h', 'wind_speed_10m_rolling_std_24h',
       'temp_wind_interaction', 'cloud_radiation_interaction',
       'temperature_2m_change', 'cloud_cover_change', 'wind_speed_10m_change',
       'weather_stability_index']

    X_new_test = new_test[standardize_predictor_list]
    predictor_scaler = joblib.load('predictor_scaler_fit.pkl')
    X_new_test= predictor_scaler.transform(X_new_test)
    
    new_stand_test = pd.DataFrame(X_new_test, index=new_test[standardize_predictor_list].index, columns=new_test[standardize_predictor_list].columns) 

    categorical_columns = ['time_interval']

    encoder = joblib.load('encoded_features.pkl')    
    encoded_features_test = encoder.transform(new_test[categorical_columns])
    encoded_test = pd.DataFrame(encoded_features_test, columns=categorical_columns, index=new_test.index)

    new_stand_test = pd.concat([new_stand_test, encoded_test], axis = 1)

    return new_stand_test

def main():

    st.title('Solar Power Generation Forecasting')
    st.write('Using Open-Meteo API for weather data')
    
    # Yulara coordinates (default)
    lat = -25.2406
    lon = 130.9889
    altitude = 492
    
    # Get valid date range
    min_date, max_date = get_valid_date_range()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Forecast Configuration")
        
        # # Location input
        # st.subheader("Location Settings")
        # lat = st.number_input('Latitude', value=default_lat, format="%.4f")
        # lon = st.number_input('Longitude', value=default_lon, format="%.4f")
        
        # Date range selection
        st.subheader("Date Range Selection")
        start_date = st.date_input(
            "Start Date",
            min_value=min_date,
            max_value=max_date,
            value=min_date
        )
        
        end_date = st.date_input(
            "End Date",
            min_value=start_date,
            max_value=max_date,
            value=min(start_date + timedelta(days=2), max_date)
        )
        
        if start_date > end_date:
            st.error("End date must be after start date")
            return
        
        # Add a date range validator
        date_diff = (end_date - start_date).days
        if date_diff > 14:
            st.error("Date range cannot exceed 14 days")
            return
    
    # Main content area
    st.subheader(f"Forecast for {start_date} to {end_date}")
    
    # Display selected parameters
    st.write(f"""
    **Selected Parameters:**
    - Location: {lat:.4f}°N, {lon:.4f}°E
    - Date Range: {start_date} to {end_date} ({date_diff + 1} days)
    """)
    
    # Fetch weather data button
    if st.button('Generate Forecast'):
        with st.spinner('Fetching weather data and generating forecast...'):
            weather_data = fetch_weather_forecast(lat, lon, start_date, end_date)
            
            if weather_data:
                # Process weather data
                df = process_weather_data(weather_data)

                df['hour_of_day'] = df['date'].dt.hour

                # Filter for daylight hours (6 AM to 7 PM)
                df_daylight = df[df['hour_of_day'].between(6, 19)].copy()
                
                # Load model and make predictions
                try:
                    model = load_model()
                    
                    # Prepare features for prediction
                    features = add_engineered_features(df_daylight,start_date, lat, lon , altitude)

                    features_std = standardize_data(features)

                    dtest = xgb.DMatrix(features_std)
                                        
                    # Make predictions
                    predictions = model.predict(dtest)

                    df_daylight = df_daylight[(df_daylight['date'] >= pd.Timestamp(start_date))]
                    
                    df_daylight['predicted_power'] = predictions
                    
                    # Create tabs for different views
                    tab1, tab2, tab3 = st.tabs(["Forecast Plot", "Daily Summary", "Hourly Data"])
                    
                    with tab1:
                        # Create plot
                        fig = go.Figure()
                        
                        # Add power prediction line only
                        fig.add_trace(go.Scatter(
                            x=df_daylight['date'],
                            y=df_daylight['predicted_power'],
                            name='Predicted Power',
                            line=dict(color='blue')
                        ))
                        
                        fig.update_layout(
                            title='Solar Power Generation Forecast',
                            xaxis_title='Time',
                            yaxis_title='Power (kW)',
                            height=600
                        )
                        
                        st.plotly_chart(fig)
                    
                    with tab2:
                        # Daily summary statistics
                        daily_summary = df_daylight.groupby(df_daylight['date'].dt.date).agg({
                            'predicted_power': ['mean', 'max', 'min', 'sum'],
                            'temperature_2m': ['mean', 'max', 'min'],
                            'cloud_cover': 'mean'
                        }).round(2)
                        
                        daily_summary.columns = ['Avg Power (kW)', 'Max Power (kW)', 
                                               'Min Power (kW)', 'Total Energy (kWh)',
                                               'Avg Temp (°C)', 'Max Temp (°C)', 
                                               'Min Temp (°C)', 'Avg Cloud Cover (%)']
                        
                        st.write("Daily Summary Statistics")
                        st.dataframe(daily_summary)
                    
                    with tab3:
                        # Hourly forecast data
                        st.write("Hourly Forecast Data")
                        display_cols = ['date', 'predicted_power', 'temperature_2m', 
                                      'cloud_cover', 'direct_radiation']
                        
                        # Format the date column
                        hourly_data = df_daylight[display_cols].copy()
                        hourly_data['date'] = hourly_data['date'].dt.strftime('%Y-%m-%d %H:%M')
                        
                        st.dataframe(hourly_data.set_index('date'))
                    
                    # Add download buttons
                    csv = df_daylight.to_csv(index=False)
                    st.download_button(
                        label="Download Full Forecast Data",
                        data=csv,
                        file_name=f"solar_forecast_{start_date}_to_{end_date}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f'Error making prediction: {str(e)}')
            else:
                st.error('Failed to fetch weather data. Please try again.')

if __name__ == '__main__':
    main()