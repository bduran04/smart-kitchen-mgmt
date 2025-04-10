import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings
import json
import multiprocessing
from functools import partial
import itertools
from db_utils import connect_to_db
from fetch_menu_ingredients import fetch_menu_items_ingredients
from fetch_orders import fetch_historical_orders

# Suppress warning messages for cleaner output
warnings.filterwarnings('ignore')

###################################################################################
# PART 1: DATA COLLECTION AND PROCESSING
###################################################################################

def calculate_ingredient_needs(start_date=None, end_date=None):
    """
    Convert order data into ingredient needs broken down by hour.
    
    This function:
    1. Fetches historical orders from the database
    2. Fetches menu items and their ingredient requirements
    3. Maps the order quantities to ingredient usage
    4. Creates a DataFrame indexed by hourly timestamps with ingredients as columns
    
    Parameters:
    -----------
    start_date : datetime.datetime, optional
        Start date for the query range. If None, defaults to 90 days ago.
    end_date : datetime.datetime, optional
        End date for the query range. If None, defaults to current time.
        
    Returns:
    --------
    dict
        Dictionary containing DataFrames with ingredient usage:
        - 'hourly': DataFrame indexed by hourly timestamps
        - 'daily': DataFrame with daily aggregated data
    """
    print("Retrieving order history...")
    orders = fetch_historical_orders(start_date, end_date)
    
    if not orders:
        print("❌ No orders found. Cannot calculate ingredient needs.")
        return {'hourly': pd.DataFrame(), 'daily': pd.DataFrame()}
    
    print(f"Retrieved {len(orders)} order items.")
    
    print("Retrieving menu item ingredients...")
    menu_items = fetch_menu_items_ingredients()
    
    if not menu_items:
        print("❌ No menu item ingredients found. Cannot calculate ingredient needs.")
        return {'hourly': pd.DataFrame(), 'daily': pd.DataFrame()}
    
    print(f"Retrieved ingredients for {len(menu_items)} menu items.")
    
    # Get list of all unique ingredients
    all_ingredients = set()
    for ingredients in menu_items.values():
        all_ingredients.update(ingredients.keys())
    print(f"Identified {len(all_ingredients)} unique ingredients.")
    
    # Create timestamp range
    # Find min and max order timestamps
    if orders:
        start_time = min(order['ordertimestamp'] for order in orders).replace(minute=0, second=0, microsecond=0)
        end_time = max(order['ordertimestamp'] for order in orders).replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    else:
        return {'hourly': pd.DataFrame(), 'daily': pd.DataFrame()}  # Return empty DataFrame if no orders
    
    # Create DataFrame with hourly timestamps
    print(f"Creating hourly ingredient needs from {start_time} to {end_time}...")
    date_range = pd.date_range(start=start_time, end=end_time, freq='H')
    ingredients_df = pd.DataFrame(index=date_range, columns=list(all_ingredients))
    ingredients_df = ingredients_df.fillna(0.0)  # Fill with float 0.0 instead of integer 0
    # Explicitly set data types to float to avoid type conversion warnings
    ingredients_df = ingredients_df.astype(float)
    
    # Process each order
    processed_orders = 0
    skipped_orders = 0
    
    for order in orders:
        # Get the hour timestamp (rounded down to the nearest hour)
        hour_timestamp = order['ordertimestamp'].replace(minute=0, second=0, microsecond=0)
        menu_item_name = order['menuitemname']
        
        # Skip if this menu item doesn't have ingredient data
        if menu_item_name not in menu_items:
            skipped_orders += 1
            continue
        
        processed_orders += 1
        
        # Process each ingredient for this menu item
        for ingredient, amount in menu_items[menu_item_name].items():
            # Accumulate the ingredient amount needed
            # Assume quantity is always 1 since we don't have that info in orderitems table
            ingredients_df.at[hour_timestamp, ingredient] += amount
    
    print(f"Processed {processed_orders} orders, skipped {skipped_orders} orders.")
    
    # Calculate some statistics for informational purposes
    total_ingredient_usage = ingredients_df.sum().sum()
    busiest_hour = ingredients_df.sum(axis=1).idxmax()
    busiest_hour_usage = ingredients_df.sum(axis=1).max()
    
    print(f"\n📊 Ingredient Usage Statistics:")
    print(f"  - Total ingredient usage: {total_ingredient_usage:.2f} units")
    print(f"  - Busiest hour: {busiest_hour} with {busiest_hour_usage:.2f} units")
    print(f"  - Average hourly usage: {total_ingredient_usage / len(date_range):.2f} units")
    
    # Get the top 5 ingredients by usage
    top_ingredients = ingredients_df.sum().sort_values(ascending=False).head(5)
    print(f"\nTop 5 ingredients by usage:")
    for ingredient, usage in top_ingredients.items():
        print(f"  - Ingredient {ingredient}: {usage:.2f} units")
    
    # Additional preprocessing for time series analysis
    # Resample to daily frequency for more stable forecasting
    daily_ingredients_df = ingredients_df.resample('D').sum()
    
    # Return both hourly and daily data for different modeling approaches
    return {
        'hourly': ingredients_df,
        'daily': daily_ingredients_df
    }


def clear_forecast_database(conn):
    """
    Clear all records from the forecasts table.
    
    Parameters:
    -----------
    conn : psycopg2.extensions.connection
        Database connection object
        
    Returns:
    --------
    bool
        True if operation was successful, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        cursor.execute("DELETE FROM forecasts")
        deleted_count = cursor.rowcount
        conn.commit()
        print(f"✅ Successfully cleared {deleted_count} records from forecasts table")
        return True
    except Exception as e:
        conn.rollback()
        print(f"❌ Error clearing forecasts table: {e}")
        return False
    finally:
        cursor.close()


def store_predictions_in_db(conn, forecasts, recommendations):
    """
    Store predictions in the database forecasts table.
    
    Parameters:
    -----------
    conn : psycopg2.extensions.connection
        Database connection object
    forecasts : dict or pandas.DataFrame
        Dictionary with forecast DataFrames or a single DataFrame
    recommendations : dict
        Dictionary with various recommendation types
    
    Returns:
    --------
    list
        List of forecast IDs that were created
    """
    import json
    import psycopg2
    from datetime import datetime, timedelta
    
    # Handle both new dict format and old DataFrame format
    if isinstance(forecasts, dict) and 'hourly' in forecasts:
        forecast_data = forecasts['hourly']
    elif isinstance(forecasts, pd.DataFrame):
        forecast_data = forecasts
    else:
        print("❌ Invalid forecast data format.")
        return []
    
    if forecast_data.empty:
        print("❌ No forecast data to store in database.")
        return []
        
    cursor = conn.cursor()
    forecast_ids = []
    current_time = datetime.now()
    current_date = current_time.date()
    
    try:
        # Only store recommendations for future dates
        if 'daily_needs' in recommendations and not recommendations['daily_needs'].empty:
            daily_df = recommendations['daily_needs']
            
            # Process each forecasted day separately, but only future dates
            for date_idx in daily_df.index:
                # Convert Timestamp to date if needed
                if hasattr(date_idx, 'date'):
                    date_idx_date = date_idx.date()
                else:
                    date_idx_date = date_idx
                
                # Skip past dates
                if date_idx_date <= current_date:
                    continue
                    
                # Create a dict of ingredient quantities for this day
                daily_prep = {}
                for col in daily_df.columns:
                    if not pd.isna(daily_df.loc[date_idx, col]) and daily_df.loc[date_idx, col] > 0:
                        daily_prep[str(col)] = float(daily_df.loc[date_idx, col])
                
                # Skip if there are no significant quantities
                if not daily_prep:
                    continue
                
                # Use the date as the recommendationfor field
                if hasattr(date_idx, 'date'):
                    target_date = date_idx.date()
                else:
                    target_date = date_idx
                
                # Store in database - just the ingredient quantities for clarity
                query = """
                INSERT INTO forecasts (recommendation, recommendationfor, createdat) 
                VALUES (%s, %s, %s)
                RETURNING forecastid
                """
                
                cursor.execute(query, (json.dumps(daily_prep), target_date, current_time))
                forecast_id = cursor.fetchone()[0]
                forecast_ids.append(forecast_id)
                print(f"✅ Stored prep recommendation for {target_date.strftime('%Y-%m-%d')} with ID: {forecast_id}")
        
        # Commit the transaction
        conn.commit()
        print(f"✅ Successfully stored recommendations for future dates in database")
        return forecast_ids
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error storing forecast in database: {e}")
        return []
        
    finally:
        cursor.close()


def store_traffic_recommendations(conn, traffic_recommendations):
    """
    Store traffic recommendations in the database.
    
    Parameters:
    -----------
    conn : psycopg2.extensions.connection
        Database connection object
    traffic_recommendations : dict
        Dictionary of dates with text recommendations
        
    Returns:
    --------
    list
        List of forecast IDs that were created
    """
    import json
    from datetime import datetime
    
    if not traffic_recommendations:
        print("❌ No traffic recommendations to store.")
        return []
        
    cursor = conn.cursor()
    forecast_ids = []
    current_time = datetime.now()
    
    try:
        for date, recommendation in traffic_recommendations.items():
            query = """
            INSERT INTO forecasts (recommendation, recommendationfor, createdat) 
            VALUES (%s, %s, %s)
            RETURNING forecastid
            """
            
            # Package the recommendation in a JSON object for clarity
            recommendation_data = json.dumps({
                "type": "traffic_forecast",
                "text_recommendation": recommendation
            })
            
            cursor.execute(query, (recommendation_data, date, current_time))
            forecast_id = cursor.fetchone()[0]
            forecast_ids.append(forecast_id)
            print(f"✅ Stored traffic recommendation for {date.strftime('%Y-%m-%d')} with ID: {forecast_id}")
        
        conn.commit()
        return forecast_ids
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error storing traffic recommendations: {e}")
        return []
        
    finally:
        cursor.close()


###################################################################################
# PART 2: FORECASTING WITH SARIMAX MODELS
###################################################################################

def forecast_future_needs(ingredients_data, future_hours=24*7):  # Default to 1 week forecast
    """
    Generate ingredient forecast for future hours using SARIMAX models.
    
    Parameters:
    -----------
    ingredients_data : dict or pandas.DataFrame
        Dictionary with hourly and daily historical ingredient usage data
        or DataFrame with hourly data (for backward compatibility)
    future_hours : int, optional
        Number of hours to forecast (default: 1 week)
        
    Returns:
    --------
    dict
        Dictionary with hourly and daily forecast DataFrames
    """
    # Convert old format to new format if needed
    if isinstance(ingredients_data, pd.DataFrame):
        print("⚠️ Old format detected - converting to new dictionary format")
        ingredients_data = {
            'hourly': ingredients_data,
            'daily': ingredients_data.resample('D').sum()
        }
    
    if ingredients_data['hourly'].empty:
        print("❌ No historical data available for forecasting.")
        return {'hourly': pd.DataFrame(), 'daily': pd.DataFrame()}
    
    # Create date range for future hours
    current_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    future_hourly_dates = pd.date_range(start=current_time, periods=future_hours, freq='H')
    
    # Calculate number of days to forecast
    future_days = (future_hours + 23) // 24  # Round up to the nearest day
    future_daily_dates = pd.date_range(start=current_time.date(), periods=future_days, freq='D')
    
    # Initialize forecast DataFrames
    hourly_forecasts = pd.DataFrame(index=future_hourly_dates, columns=ingredients_data['hourly'].columns)
    daily_forecasts = pd.DataFrame(index=future_daily_dates, columns=ingredients_data['daily'].columns)
    
    hourly_forecasts = hourly_forecasts.fillna(0.0)
    daily_forecasts = daily_forecasts.fillna(0.0)
    
    print(f"\n🔮 Forecasting ingredient needs for the next {future_hours} hours ({future_days} days)...")
    
    # Identify ingredients with sufficient data for SARIMAX modeling
    valid_ingredients = _identify_forecastable_ingredients(ingredients_data)
    
    if not valid_ingredients:
        print("⚠️ No ingredients with sufficient data for SARIMAX modeling.")
        print("Using fallback method for all ingredients.")
        return _forecast_with_fallback(ingredients_data, future_hours)
    
    print(f"🧪 Found {len(valid_ingredients)} ingredients with sufficient data for SARIMAX modeling")
    
    # Get all ingredients that have any historical data
    all_used_ingredients = [
        col for col in ingredients_data['daily'].columns 
        if ingredients_data['daily'][col].sum() > 0
    ]
    
    # If we only have a few valid ingredients for SARIMAX, try to forecast more with fallback
    # methods first to ensure we have predictions for all important ingredients
    if len(valid_ingredients) < len(all_used_ingredients) / 2:
        print(f"⚠️ Only {len(valid_ingredients)} of {len(all_used_ingredients)} ingredients can use SARIMAX.")
        print("Running fallback forecasting for all ingredients first...")
        
        # Get fallback forecasts for all ingredients
        fallback_forecasts = _forecast_with_fallback(ingredients_data, future_hours)
        
        # We'll use these as a base and replace with SARIMAX where possible
        hourly_forecasts = fallback_forecasts['hourly'].copy()
        daily_forecasts = fallback_forecasts['daily'].copy()
    
    # Use multiprocessing to speed up model fitting
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    print(f"🖥️ Using {num_cores} CPU cores for parallel processing")
    
    # Process daily forecasts first (these are more stable)
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Partial function with fixed parameters
        forecast_func = partial(
            _forecast_ingredient_daily,
            historical_data=ingredients_data['daily'],
            future_dates=future_daily_dates
        )
        
        # Try to model ALL valid ingredients with SARIMAX, not just a sample
        # This ensures we get predictions for as many ingredients as possible
        
        # Map the function to the ingredients
        results = pool.map(forecast_func, valid_ingredients)
        
        # Update the forecasts DataFrame with results
        for ingredient, forecast_values in results:
            if ingredient in daily_forecasts.columns:
                daily_forecasts[ingredient] = forecast_values
    
    # Handle hourly forecasts for all ingredients that had successful SARIMAX daily forecasts
    ingredients_with_forecasts = [
        col for col in daily_forecasts.columns 
        if daily_forecasts[col].sum() > 0
    ]
    
    if ingredients_with_forecasts:
        print(f"⏱️ Generating detailed hourly forecasts for {len(ingredients_with_forecasts)} ingredients")
        with multiprocessing.Pool(processes=num_cores) as pool:
            # Partial function with fixed parameters
            hourly_forecast_func = partial(
                _forecast_ingredient_hourly,
                historical_data=ingredients_data['hourly'],
                daily_forecast=daily_forecasts,
                future_dates=future_hourly_dates
            )
            
            # Map the function to all ingredients with daily forecasts
            hourly_results = pool.map(hourly_forecast_func, ingredients_with_forecasts)
            
            # Update the forecasts DataFrame with results
            for ingredient, forecast_values in hourly_results:
                if ingredient in hourly_forecasts.columns:
                    hourly_forecasts[ingredient] = forecast_values
    
    # Validate that we have meaningful forecasts
    # If any key ingredients have no forecast, use fallback
    top_ingredients = ingredients_data['daily'].sum().nlargest(5).index.tolist()
    
    for ingredient in top_ingredients:
        if hourly_forecasts[ingredient].sum() < 0.1 and ingredients_data['hourly'][ingredient].sum() > 0:
            print(f"⚠️ Important ingredient {ingredient} has insufficient forecast, applying fallback...")
            # Use fallback for this ingredient
            ingredient_fallback = _fallback_forecast_daily(ingredient, ingredients_data['daily'], future_daily_dates)
            daily_forecasts[ingredient] = ingredient_fallback[1]
            
            # Distribute to hourly
            hourly_forecasts[ingredient] = _distribute_daily_to_hourly(
                ingredient, 
                daily_forecasts, 
                ingredients_data['hourly'],
                future_hourly_dates
            )
    
    print(f"✅ Successfully generated forecasts for {len(ingredients_data['hourly'].columns)} ingredients")
    
    return {
        'hourly': hourly_forecasts,
        'daily': daily_forecasts
    }


def _identify_forecastable_ingredients(ingredients_data):
    """
    Identify ingredients with sufficient data for SARIMAX modeling.
    
    Parameters:
    -----------
    ingredients_data : dict
        Dictionary containing DataFrames with ingredient usage data
        
    Returns:
    --------
    list
        List of ingredient names that have sufficient data
    """
    daily_data = ingredients_data['daily']
    
    valid_ingredients = []
    
    for ingredient in daily_data.columns:
        series = daily_data[ingredient]
        
        # Check if there's enough non-zero data points
        non_zero_count = (series > 0).sum()
        
        # Lowered threshold - need at least 7 days of data for SARIMAX
        # This makes the model applicable to more ingredients
        if non_zero_count >= 7:
            # Check if there's a pattern (non-random) by looking at autocorrelation
            # Will accept more ingredients as having patterns
            valid_ingredients.append(ingredient)
    
    return valid_ingredients


def _has_pattern(series):
    """
    Check if a time series has a pattern (is not random).
    For our use case, we'll assume most ingredient usage patterns 
    are non-random to allow more ingredients to be modeled with SARIMAX.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
        
    Returns:
    --------
    bool
        True if the series appears to have a pattern, False otherwise
    """
    # Need at least 3 non-zero values
    if (series > 0).sum() < 3:
        return False
    
    # For restaurant ingredients, assume there's a pattern if there's sufficient data
    # This is a reasonable assumption since restaurant orders typically follow
    # patterns based on days of the week, time of day, etc.
    return True


def _find_best_sarimax_parameters(series):
    """
    Find the best SARIMAX parameters for a given time series.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
        
    Returns:
    --------
    tuple
        Best parameters (p, d, q, P, D, Q, s)
    """
    best_aic = float('inf')
    best_params = None
    
    # Define parameter grid for SARIMAX
    # Keep it simple to avoid excessive computation time
    p = d = q = range(0, 2)
    P = D = Q = range(0, 2)
    s = [7]  # Weekly seasonality
    
    # Generate all combinations of parameters
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = list(itertools.product(P, D, Q, s))
    
    # Sample a subset of combinations to reduce computation time
    # Select a maximum of 4 combinations to try
    if len(pdq) > 2:
        import random
        random.seed(42)  # For reproducibility
        pdq = random.sample(pdq, 2)
    
    if len(seasonal_pdq) > 2:
        import random
        random.seed(42)  # For reproducibility
        seasonal_pdq = random.sample(seasonal_pdq, 2)
    
    # Try each combination
    for param in pdq:
        for seasonal_param in seasonal_pdq:
            try:
                # Fit the model
                mod = SARIMAX(series,
                            order=param,
                            seasonal_order=seasonal_param,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
                
                results = mod.fit(disp=False, maxiter=50)
                
                # If this is the best model so far, save the parameters
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (param, seasonal_param)
            except:
                continue
    
    # If no valid model found, use default parameters
    if best_params is None:
        best_params = ((1, 0, 1), (1, 0, 1, 7))
    
    # Unpack parameters for return
    p, d, q = best_params[0]
    P, D, Q, s = best_params[1]
    
    return p, d, q, P, D, Q, s


def _forecast_ingredient_daily(ingredient, historical_data, future_dates):
    """
    Forecast daily usage for a single ingredient using SARIMAX.
    
    Parameters:
    -----------
    ingredient : str
        Name of the ingredient to forecast
    historical_data : pandas.DataFrame
        Historical daily ingredient usage
    future_dates : pandas.DatetimeIndex
        Dates to forecast for
        
    Returns:
    --------
    tuple
        (ingredient name, forecasted values)
    """
    try:
        # Extract the series for this ingredient
        series = historical_data[ingredient]
        
        # Check if there's enough data - now more lenient
        if (series > 0).sum() < 7:
            raise ValueError("Insufficient data")
        
        # Find best parameters
        p, d, q, P, D, Q, s = _find_best_sarimax_parameters(series)
        
        # Fit SARIMAX model with the best parameters
        model = SARIMAX(
            series,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        model_fit = model.fit(disp=False, maxiter=100)  # Increased iterations for better convergence
        
        # Generate forecast
        forecast = model_fit.get_forecast(steps=len(future_dates))
        forecast_values = forecast.predicted_mean
        
        # Adjust index to match the future dates
        forecast_values.index = future_dates
        
        # Ensure forecast doesn't include negative values
        forecast_values = forecast_values.clip(lower=0)
        
        # Apply a scaling factor if the ingredient has very little forecasted usage
        if forecast_values.sum() < 0.1 and series.sum() > 0:
            # If forecast is almost zero but historical data shows usage,
            # scale up slightly based on historical average
            daily_avg = series[series > 0].mean()
            if not np.isnan(daily_avg):
                forecast_values = forecast_values + (daily_avg * 0.1)  # Add 10% of daily average
        
        print(f"  ✅ SARIMAX forecast completed for {ingredient} (daily)")
        return (ingredient, forecast_values)
    
    except Exception as e:
        print(f"  ⚠️ SARIMAX forecast failed for {ingredient} (daily): {str(e)}")
        
        # Fall back to a simpler method
        return _fallback_forecast_daily(ingredient, historical_data, future_dates)


def _forecast_ingredient_hourly(ingredient, historical_data, daily_forecast, future_dates):
    """
    Forecast hourly usage for a single ingredient.
    Uses both SARIMAX for overall trend and pattern detection,
    then distributes according to historical hourly patterns.
    
    Parameters:
    -----------
    ingredient : str
        Name of the ingredient to forecast
    historical_data : pandas.DataFrame
        Historical hourly ingredient usage
    daily_forecast : pandas.DataFrame
        Daily forecasts from SARIMAX
    future_dates : pandas.DatetimeIndex
        Hours to forecast for
        
    Returns:
    --------
    tuple
        (ingredient name, forecasted values)
    """
    try:
        # Initialize forecast Series
        forecast_values = pd.Series(index=future_dates, data=0.0)
        
        # The goal is to distribute the daily forecasts according to hourly patterns
        # First, find the hourly pattern
        series = historical_data[ingredient]
        
        # Get hour of day factors
        hour_factors = _calculate_hour_factors(series)
        
        # For each future day, distribute the daily forecast according to hour factors
        for day in pd.date_range(future_dates[0].date(), future_dates[-1].date(), freq='D'):
            # Skip if this day is not in the daily forecast
            if day not in daily_forecast.index:
                continue
            
            # Get the daily forecast for this day
            daily_amount = daily_forecast.at[day, ingredient]
            
            # Distribute to hours in this day
            day_hours = [dt for dt in future_dates if dt.date() == day.date()]
            
            for hour_dt in day_hours:
                hour = hour_dt.hour
                # Use the hour factor to distribute
                if hour in hour_factors:
                    forecast_values[hour_dt] = daily_amount * hour_factors[hour]
        
        print(f"  ✅ Hourly distribution completed for {ingredient}")
        return (ingredient, forecast_values)
    
    except Exception as e:
        print(f"  ⚠️ Hourly forecast failed for {ingredient}: {str(e)}")
        
        # Fall back to simple distribution
        forecast_values = _distribute_daily_to_hourly(
            ingredient, 
            daily_forecast, 
            historical_data,
            future_dates
        )
        
        return (ingredient, forecast_values)


def _calculate_hour_factors(series):
    """
    Calculate hourly distribution factors from historical data.
    
    Parameters:
    -----------
    series : pandas.Series
        Historical hourly data for an ingredient
        
    Returns:
    --------
    dict
        Dictionary mapping hours to their relative weights
    """
    # Group by hour and calculate average
    hour_avgs = series.groupby(series.index.hour).mean()
    
    # Calculate total to get proportions
    total = hour_avgs.sum()
    
    # Convert to dictionary of factors
    if total > 0:
        hour_factors = {hour: avg / total for hour, avg in hour_avgs.items()}
    else:
        # If no historical data, use uniform distribution
        hour_factors = {hour: 1/24 for hour in range(24)}
    
    return hour_factors


def _distribute_daily_to_hourly(ingredient, daily_forecast, historical_hourly, future_dates):
    """
    Distribute daily forecasts to hourly based on historical patterns.
    
    Parameters:
    -----------
    ingredient : str
        Name of the ingredient
    daily_forecast : pandas.DataFrame
        DataFrame with daily forecasts
    historical_hourly : pandas.DataFrame
        Historical hourly data
    future_dates : pandas.DatetimeIndex
        Future hourly timestamps
        
    Returns:
    --------
    pandas.Series
        Hourly forecast values
    """
    # Calculate hourly factors
    series = historical_hourly[ingredient]
    hour_factors = _calculate_hour_factors(series)
    
    # Initialize forecast Series
    forecast_values = pd.Series(index=future_dates, data=0.0)
    
    # For each future day, distribute the daily forecast
    for day in pd.date_range(future_dates[0].date(), future_dates[-1].date(), freq='D'):
        # Skip if day is not in daily forecast
        if day not in daily_forecast.index:
            continue
        
        # Get daily amount
        daily_amount = daily_forecast.at[day, ingredient]
        
        # Get hours in this day
        day_hours = [dt for dt in future_dates if dt.date() == day.date()]
        
        # Distribute amount across hours
        for hour_dt in day_hours:
            hour = hour_dt.hour
            # Use the hour factor to distribute
            if hour in hour_factors:
                forecast_values[hour_dt] = daily_amount * hour_factors[hour]
    
    return forecast_values


def _fallback_forecast_daily(ingredient, historical_data, future_dates):
    """
    Fallback method for daily forecasting when SARIMAX fails.
    Uses a more advanced average approach that considers recent trends.
    
    Parameters:
    -----------
    ingredient : str
        Name of the ingredient
    historical_data : pandas.DataFrame
        Historical daily data
    future_dates : pandas.DatetimeIndex
        Future dates to forecast for
        
    Returns:
    --------
    tuple
        (ingredient name, forecasted values)
    """
    # Initialize forecast Series
    forecast_values = pd.Series(index=future_dates, data=0.0)
    
    # Extract the series for this ingredient
    series = historical_data[ingredient]
    
    # Calculate average by day of week
    day_of_week_avg = series.groupby(series.index.dayofweek).mean()
    
    # Make sure we have values for all days
    for day in range(7):
        if day not in day_of_week_avg.index:
            day_of_week_avg[day] = series.mean() if series.mean() > 0 else 0
    
    # Calculate a recent trend factor (last 14 days vs overall average)
    recent_days = 14
    if len(series) >= recent_days:
        recent_data = series.iloc[-recent_days:]
        recent_avg = recent_data.mean()
        overall_avg = series.mean()
        
        # Avoid division by zero
        if overall_avg > 0:
            trend_factor = recent_avg / overall_avg
            # Limit the trend factor to reasonable bounds
            trend_factor = max(0.5, min(2.0, trend_factor))
        else:
            trend_factor = 1.0
    else:
        trend_factor = 1.0
    
    # Use day of week average for forecast, adjusted by trend
    for date in future_dates:
        day_of_week = date.dayofweek
        base_forecast = day_of_week_avg[day_of_week]
        # Apply trend factor
        forecast_values[date] = base_forecast * trend_factor
    
    # Ensure no negative values
    forecast_values = forecast_values.clip(lower=0)
    
    # Ensure we have at least some minimal forecast if there's historical usage
    min_historical = series[series > 0].min() if not series[series > 0].empty else 0
    if forecast_values.sum() < 0.1 and min_historical > 0:
        # Add a small amount based on minimum historical usage
        for date in future_dates:
            day_of_week = date.dayofweek
            if day_of_week in [4, 5, 6]:  # Weekend days + Friday
                forecast_values[date] += min_historical * 0.2  # 20% of minimum
            else:
                forecast_values[date] += min_historical * 0.1  # 10% of minimum
    
    print(f"  ✅ Enhanced fallback forecast completed for {ingredient} (daily)")
    return (ingredient, forecast_values)


def _forecast_with_fallback(ingredients_data, future_hours):
    """
    Generate forecasts using fallback methods when SARIMAX is not suitable.
    
    Parameters:
    -----------
    ingredients_data : dict
        Dictionary with historical ingredient data
    future_hours : int
        Number of hours to forecast
        
    Returns:
    --------
    dict
        Dictionary with hourly and daily forecasts
    """
    # Create date ranges
    current_time = datetime.datetime.now().replace(minute=0, second=0, microsecond=0)
    future_hourly_dates = pd.date_range(start=current_time, periods=future_hours, freq='H')
    
    future_days = (future_hours + 23) // 24  # Round up to nearest day
    future_daily_dates = pd.date_range(start=current_time.date(), periods=future_days, freq='D')
    
    # Initialize forecast DataFrames
    hourly_forecasts = pd.DataFrame(index=future_hourly_dates, columns=ingredients_data['hourly'].columns)
    daily_forecasts = pd.DataFrame(index=future_daily_dates, columns=ingredients_data['daily'].columns)
    
    hourly_forecasts = hourly_forecasts.fillna(0.0)
    daily_forecasts = daily_forecasts.fillna(0.0)
    
    print("Using simple averaging method for forecasting...")
    
    # Process each ingredient
    for ingredient in ingredients_data['daily'].columns:
        # Skip if no usage
        if ingredients_data['daily'][ingredient].sum() == 0:
            continue
            
        # Calculate average by day of week
        ingredient_series = ingredients_data['daily'][ingredient]
        day_of_week_avg = ingredient_series.groupby(ingredient_series.index.dayofweek).mean()
        
        # Calculate average by hour of day (for hourly distribution)
        hourly_series = ingredients_data['hourly'][ingredient]
        hour_of_day_avg = hourly_series.groupby(hourly_series.index.hour).mean()
        
        # Make sure we have values for all days and hours
        for day in range(7):
            if day not in day_of_week_avg.index:
                day_of_week_avg[day] = ingredient_series.mean() if ingredient_series.mean() > 0 else 0
                
        for hour in range(24):
            if hour not in hour_of_day_avg.index:
                hour_of_day_avg[hour] = hourly_series.mean() if hourly_series.mean() > 0 else 0
        
        # Calculate the overall average
        overall_mean = ingredient_series.mean()
        
        # Generate daily forecasts
        for future_date in future_daily_dates:
            # Use day of week average
            day_factor = day_of_week_avg[future_date.dayofweek] / overall_mean if overall_mean > 0 else 0
            daily_forecasts.at[future_date, ingredient] = day_factor * overall_mean
        
        # Generate hourly forecasts based on daily totals and hour distribution
        hour_factors = _calculate_hour_factors(hourly_series)
        
        # Distribute daily forecasts to hours
        for day in future_daily_dates:
            day_total = daily_forecasts.at[day, ingredient]
            day_hours = [dt for dt in future_hourly_dates if dt.date() == day.date()]
            
            for hour_dt in day_hours:
                hour = hour_dt.hour
                if hour in hour_factors:
                    hourly_forecasts.at[hour_dt, ingredient] = day_total * hour_factors[hour]
    
    print("✅ Completed fallback forecasting for all ingredients")
    
    return {
        'hourly': hourly_forecasts,
        'daily': daily_forecasts
    }


###################################################################################
# PART 3: ANALYSIS AND RECOMMENDATION GENERATION
###################################################################################

def analyze_forecast(forecasts):
    """
    Analyze the forecasted ingredient needs to provide useful insights.

    Parameters:
    -----------
    forecasts : dict or pandas.DataFrame
        Dictionary with forecast DataFrames or a single DataFrame
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    # Handle both new dict format and old DataFrame format
    if isinstance(forecasts, dict) and 'hourly' in forecasts:
        hourly_data = forecasts['hourly']
        daily_data = forecasts['daily']
    elif isinstance(forecasts, pd.DataFrame):
        hourly_data = forecasts
        daily_data = forecasts.resample('D').sum()
    else:
        print("❌ Invalid forecast data format.")
        return {}
    
    if hourly_data.empty:
        print("❌ No forecast data to analyze.")
        return {}
    
    print("\n📊 Forecast Analysis:")
    
    # Total forecasted amounts
    total_by_ingredient = hourly_data.sum()
    print("\nTotal forecasted amounts:")
    for ingredient, total in total_by_ingredient.nlargest(10).items():
        print(f"  - {ingredient}: {total:.2f} units")
    
    # Identify peak hours
    hourly_totals = hourly_data.sum(axis=1)
    peak_hour = hourly_totals.idxmax()
    peak_usage = hourly_totals.max()
    
    print(f"\nBusiest hour: {peak_hour.strftime('%Y-%m-%d %H:%M')} with {peak_usage:.2f} total units")
    
    # Create a copy of forecasts for analysis
    hourly_copy = hourly_data.copy()
    
    # Analyze by day of week
    hourly_copy['day_of_week'] = hourly_copy.index.day_name()
    day_of_week_totals = hourly_copy.groupby('day_of_week').sum()
    
    # Remove the day_of_week column from the groupby result if it exists
    if 'day_of_week' in day_of_week_totals.columns:
        day_of_week_totals = day_of_week_totals.drop(columns=['day_of_week'])
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Only reorder if all days are present
    if all(day in day_of_week_totals.index for day in days_order):
        day_of_week_totals = day_of_week_totals.reindex(days_order)
    
    print("\nIngredient needs by day of week:")
    for day, row in day_of_week_totals.iterrows():
        print(f"  - {day}: {row.sum():.2f} total units")
    
    # Analyze by hour of day
    hourly_copy['hour'] = hourly_copy.index.hour
    hour_of_day_totals = hourly_copy.groupby('hour').sum()
    
    # Remove the non-ingredient columns from the groupby result if they exist
    non_ingredient_cols = ['day_of_week', 'hour']
    cols_to_drop = [col for col in non_ingredient_cols if col in hour_of_day_totals.columns]
    if cols_to_drop:
        hour_of_day_totals = hour_of_day_totals.drop(columns=cols_to_drop)
    
    print("\nBusiest hours of the day:")
    for hour, total in hour_of_day_totals.sum(axis=1).nlargest(5).items():
        print(f"  - {hour:02d}:00: {total:.2f} total units")
    
    # Analyze confidence intervals for key ingredients if using SARIMAX models
    # For demo purposes, we're not calculating actual confidence intervals here
    # In a real implementation, we would use the prediction intervals from SARIMAX
    
    # Return analysis results
    return {
        'total_by_ingredient': total_by_ingredient,
        'hourly_totals': hourly_totals,
        'day_of_week_totals': day_of_week_totals,
        'hour_of_day_totals': hour_of_day_totals
    }


def prepare_recommendations(forecasts):
    """
    Generate practical recommendations based on the forecast.
    
    Parameters:
    -----------
    forecasts : dict or pandas.DataFrame
        Dictionary with forecast DataFrames or a single DataFrame
        
    Returns:
    --------
    dict
        Dictionary with various recommendation types
    """
    # Handle both new dict format and old DataFrame format
    if isinstance(forecasts, dict) and 'daily' in forecasts:
        daily_forecast = forecasts['daily']
    elif isinstance(forecasts, pd.DataFrame):
        daily_forecast = forecasts.resample('D').sum()
    else:
        print("❌ Invalid forecast data format.")
        return {}
    
    if daily_forecast.empty:
        print("❌ No forecast data for recommendations.")
        return {}
    
    recommendations = {}
    
    # Calculate daily needs with 15% safety margin
    daily_with_margin = daily_forecast.copy()
    for col in daily_with_margin.columns:
        daily_with_margin[col] = daily_with_margin[col] * 1.15
    
    recommendations['daily_needs'] = daily_with_margin
    
    # Get tomorrow's needs
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    if tomorrow in daily_with_margin.index:
        recommendations['tomorrow_needs'] = daily_with_margin.loc[tomorrow]
    
    # Calculate top ingredients by day
    top_daily_ingredients = {}
    for date, row in daily_forecast.iterrows():
        if not row.empty:
            top_ingredients = row.nlargest(min(5, len(row)))
            top_daily_ingredients[date.strftime('%Y-%m-%d')] = {
                ingredient: round(amount, 2) for ingredient, amount in top_ingredients.items() if amount > 0
            }
    
    recommendations['top_ingredients_by_day'] = top_daily_ingredients
    
    print("\n📋 Preparation Recommendations:")
    
    # Print recommendations for tomorrow
    if 'tomorrow_needs' in recommendations:
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')
        print(f"\nRecommended prep quantities for tomorrow ({tomorrow_str}):")
        for ingredient, amount in recommendations['tomorrow_needs'].nlargest(10).items():
            if amount > 0:  # Only show non-zero amounts
                print(f"  - {ingredient}: {amount:.2f} units (includes 15% safety margin)")
    
    # Calculate scheduling recommendations based on busy hours
    if isinstance(forecasts, dict) and 'hourly' in forecasts:
        hourly_data = forecasts['hourly']
        
        # Find busiest hours for each day
        busy_hour_recs = {}
        
        for day in pd.date_range(hourly_data.index[0].date(), hourly_data.index[-1].date(), freq='D'):
            # Get data for this day
            day_data = hourly_data[hourly_data.index.date == day.date()]
            
            if not day_data.empty:
                # Calculate total by hour
                hourly_sum = day_data.sum(axis=1)
                
                # Get top 3 busiest hours
                busiest_hours = hourly_sum.nlargest(3)
                
                # Save recommendation
                busy_hour_recs[day.strftime('%Y-%m-%d')] = {
                    hour.strftime('%H:%M'): round(float(amount), 2) 
                    for hour, amount in busiest_hours.items()
                }
        
        recommendations['busiest_hours'] = busy_hour_recs
        
        # Print sample of busiest hours
        if busy_hour_recs:
            print("\nSample of busiest hours by day:")
            for i, (date, hours) in enumerate(list(busy_hour_recs.items())[:3]):
                print(f"  - {date}: {', '.join([f'{hour} ({amount:.2f} units)' for hour, amount in hours.items()])}")
    
    return recommendations


def generate_traffic_recommendations(ingredients_df, forecasts):
    """
    Generate textual recommendations about anticipated restaurant traffic
    based on historical data and forecasts.
    
    Parameters:
    -----------
    ingredients_df : dict or pandas.DataFrame
        Dictionary with historical ingredient usage data or a single DataFrame
    forecasts : dict or pandas.DataFrame
        Dictionary with forecasted ingredient usage or a single DataFrame
        
    Returns:
    --------
    dict
        Dictionary of dates with text recommendations for each day
    """
    # Handle both new dict format and old DataFrame format
    if isinstance(ingredients_df, dict) and 'hourly' in ingredients_df:
        historical_hourly = ingredients_df['hourly'].sum(axis=1)
    elif isinstance(ingredients_df, pd.DataFrame):
        historical_hourly = ingredients_df.sum(axis=1)
    else:
        print("❌ Invalid historical data format.")
        return {}
    
    if isinstance(forecasts, dict) and 'hourly' in forecasts:
        forecasts_copy = forecasts['hourly'].copy()
    elif isinstance(forecasts, pd.DataFrame):
        forecasts_copy = forecasts.copy()
    else:
        print("❌ Invalid forecast data format.")
        return {}
    
    # Initialize recommendations dictionary
    traffic_recommendations = {}
    
    # Create aggregated historical data by day of week and hour
    historical_hourly.index = pd.DatetimeIndex(historical_hourly.index)
    
    # Calculate day of week patterns
    day_of_week_pattern = historical_hourly.groupby(historical_hourly.index.dayofweek).mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_pattern.index = day_names
    
    # Calculate hourly patterns
    hour_pattern = historical_hourly.groupby(historical_hourly.index.hour).mean()
    
    # Identify busy days (>20% above average)
    avg_daily = day_of_week_pattern.mean()
    busy_days = [day for day, value in day_of_week_pattern.items() if value > avg_daily * 1.2]
    slow_days = [day for day, value in day_of_week_pattern.items() if value < avg_daily * 0.8]
    
    # Identify busy hours (>30% above average)
    avg_hourly = hour_pattern.mean()
    busy_hours = [hour for hour, value in hour_pattern.items() if value > avg_hourly * 1.3]
    peak_hour = hour_pattern.idxmax()
    
    # Process forecast data by day
    # First create a new column with the date
    forecasts_copy['date'] = forecasts_copy.index.date
    
    # Get only the ingredient columns - exclude any analysis columns
    ingredient_cols = [col for col in forecasts_copy.columns 
                      if col not in ['date', 'day_of_week', 'hour']]
    
    # Calculate total ingredient usage per day (sum only numeric ingredient columns)
    daily_total = forecasts_copy.groupby('date')[ingredient_cols].sum().sum(axis=1)
    
    # Identify unusually busy forecast days (>25% above average of forecasted days)
    avg_forecast = daily_total.mean()
    busy_forecast_days = {}
    
    for date_val, value in daily_total.items():
        # Use the date object to get day of week name
        weekday_num = date_val.weekday()
        day_name = day_names[weekday_num]  # Use our day_names list
        
        expected_value = day_of_week_pattern.get(day_name, avg_daily)
        
        # Check if this day is much busier than typical for this day of week
        if value > expected_value * 1.25:
            busy_forecast_days[date_val] = {
                'traffic_level': 'High',
                'percent_above_normal': f"{((value / expected_value) - 1) * 100:.1f}%",
                'day_of_week': day_name
            }
        elif value < expected_value * 0.75:
            busy_forecast_days[date_val] = {
                'traffic_level': 'Low',
                'percent_below_normal': f"{(1 - (value / expected_value)) * 100:.1f}%",
                'day_of_week': day_name
            }
        else:
            busy_forecast_days[date_val] = {
                'traffic_level': 'Normal',
                'day_of_week': day_name
            }
    
    # Generate text recommendations for each forecast day
    for date_val, info in busy_forecast_days.items():
        day_str = date_val.strftime('%Y-%m-%d')
        day_name = info['day_of_week']
        
        if info['traffic_level'] == 'High':
            recommendation = f"Expect HEAVY traffic on {day_name}, {day_str}. " + \
                            f"Forecasting {info['percent_above_normal']} above normal levels for a {day_name}. " + \
                            "Consider additional staffing and ingredient preparation."
                            
            if day_name in busy_days:
                recommendation += f" Note that {day_name}s are typically busy days already."
                
        elif info['traffic_level'] == 'Low':
            recommendation = f"Expect LIGHT traffic on {day_name}, {day_str}. " + \
                            f"Forecasting {info['percent_below_normal']} below normal levels for a {day_name}. " + \
                            "Consider reducing staffing and ingredient preparation."
                            
            if day_name in slow_days:
                recommendation += f" Note that {day_name}s are typically slower days already."
                
        else:
            recommendation = f"Expect NORMAL traffic on {day_name}, {day_str}. " + \
                            "Typical staffing and ingredient preparation should be sufficient."
        
        # Add hourly guidance - be careful with date filtering and summing
        # Get hourly data for this date
        day_data = forecasts_copy[forecasts_copy['date'] == date_val]
        
        if not day_data.empty:
            # Sum only ingredient columns for each hour
            hourly_sums = day_data[ingredient_cols].sum(axis=1)
            
            if not hourly_sums.empty:
                try:
                    peak_forecast_hour = hourly_sums.idxmax().hour
                    recommendation += f" Peak hours expected around {peak_forecast_hour:02d}:00."
                    
                    # Check if there are multiple busy periods
                    hourly_threshold = hourly_sums.mean() * 1.3
                    busy_periods = []
                    current_period = []
                    
                    # Convert to list of (hour, value) pairs for analysis
                    hourly_list = [(idx.hour, val) for idx, val in hourly_sums.items()]
                    hourly_list.sort()  # Sort by hour
                    
                    for hour, value in hourly_list:
                        if value > hourly_threshold:
                            current_period.append(hour)
                        elif current_period:
                            busy_periods.append(current_period)
                            current_period = []
                    
                    if current_period:
                        busy_periods.append(current_period)
                        
                    if len(busy_periods) > 1:
                        period_texts = []
                        for period in busy_periods:
                            if len(period) > 1:
                                period_texts.append(f"{period[0]:02d}:00-{period[-1]+1:02d}:00")
                            else:
                                period_texts.append(f"{period[0]:02d}:00")
                                
                        recommendation += f" Multiple busy periods expected: {', '.join(period_texts)}."
                except Exception as e:
                    # If anything goes wrong with the hourly analysis, just skip this part
                    print(f"Warning: Could not analyze hourly patterns for {day_str}: {e}")
        
        traffic_recommendations[date_val] = recommendation
        
    # Print sample of traffic recommendations
    print("\n🚦 Traffic Recommendations:")
    for i, (date, recommendation) in enumerate(list(traffic_recommendations.items())[:3]):
        print(f"\n{date.strftime('%Y-%m-%d')} ({date.strftime('%A')}):")
        print(f"  {recommendation}")
    
    if len(traffic_recommendations) > 3:
        print(f"\n  ... and {len(traffic_recommendations) - 3} more days")
    
    return traffic_recommendations


# Main execution
if __name__ == "__main__":
    # This section only runs when the script is executed directly
    
    # Print a header for the terminal output
    print("=" * 60)
    print("🍽️  SMART KITCHEN - INGREDIENT PREDICTION SYSTEM")
    print("=" * 60)
    
    # Parse command-line arguments
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict future ingredient needs using SARIMAX models')
    parser.add_argument('--days', type=int, default=90, help='Number of historical days to analyze (default: 90)')
    parser.add_argument('--forecast', type=int, default=7, help='Number of days to forecast (default: 7)')
    parser.add_argument('--start', type=str, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end', type=str, help='End date in YYYY-MM-DD format')
    parser.add_argument('--save', action='store_true', help='Save forecasts to CSV file')
    parser.add_argument('--clear-db', action='store_true', help='Clear existing forecast records before adding new ones')
    parser.add_argument('--auto-confirm', action='store_true', help='Skip confirmation prompts')
    parser.add_argument('--parallelize', action='store_true', help='Use parallel processing for SARIMAX models')
    parser.add_argument('--max-ingredients', type=int, default=0, help='Maximum number of ingredients to process with SARIMAX (0 = all)')
    
    args = parser.parse_args()
    
    # Set up date range based on arguments
    if args.start and args.end:
        try:
            start_date = datetime.datetime.strptime(args.start, '%Y-%m-%d')
            end_date = datetime.datetime.strptime(args.end, '%Y-%m-%d') + datetime.timedelta(days=1)
            print(f"📅 Using custom date range: {args.start} to {args.end}")
        except ValueError:
            print("❌ Invalid date format! Please use YYYY-MM-DD")
            sys.exit(1)
    else:
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=args.days)
        print(f"📅 Analyzing {args.days} days of historical data")
    
    # Check database connection first
    print("Testing database connection...")
    conn = connect_to_db()
    if conn is None:
        print("❌ Could not connect to database. Please check your configuration in db_utils.py")
        sys.exit(1)
    print("✅ Database connection successful")
    
    # Prompt to clear database if not specified in arguments and not auto-confirmed
    if not args.clear_db and not args.auto_confirm:
        user_input = input("Would you like to clear existing forecast records? (y/n): ")
        if user_input.lower() in ['y', 'yes']:
            args.clear_db = True
    
    # Clear database if requested
    if args.clear_db:
        if not clear_forecast_database(conn):
            print("⚠️ Warning: Failed to clear forecast database. Continuing with execution...")
    
    # Step 1: Calculate historical ingredient needs
    print("\n" + "-" * 60)
    print("STEP 1: Calculating historical ingredient needs")
    print("-" * 60)
    ingredients_data = calculate_ingredient_needs(start_date, end_date)
    
    if ingredients_data['hourly'].empty:
        print("❌ No historical ingredient data available. Exiting.")
        sys.exit(1)
    
    # Add diagnostic information
    print("\n🔍 Historical Data Diagnostics:")
    print(f"  - Non-zero values in historical data: {(ingredients_data['hourly'] > 0).sum().sum()}")
    print(f"  - Total ingredient usage sum: {ingredients_data['hourly'].sum().sum():.2f}")
    print(f"  - Unique timestamps with data: {len(ingredients_data['hourly'])}")
    print(f"  - Ingredients with non-zero usage: {(ingredients_data['hourly'].sum() > 0).sum()}")
    
    # Step 2: Forecast using SARIMAX models
    print("\n" + "-" * 60)
    print(f"STEP 2: Forecasting needs for the next {args.forecast} days using SARIMAX")
    print("-" * 60)
    forecast_hours = args.forecast * 24  # Convert days to hours
    
    # Use SARIMAX forecasting method
    forecasts = forecast_future_needs(ingredients_data, forecast_hours)
    
    if forecasts['hourly'].empty:
        print("❌ Failed to generate forecasts. Exiting.")
        sys.exit(1)
    
    # Step 3: Analyze forecasts and generate recommendations
    print("\n" + "-" * 60)
    print("STEP 3: Analyzing forecasts and generating recommendations")
    print("-" * 60)
    analysis = analyze_forecast(forecasts)
    recommendations = prepare_recommendations(forecasts)
    
    # Step 4: Generate traffic recommendations
    print("\n" + "-" * 60)
    print("STEP 4: Generating traffic recommendations")
    print("-" * 60)
    traffic_recommendations = generate_traffic_recommendations(ingredients_data, forecasts)
    
    # Step 5: Store recommendations in a database
    print("\n" + "-" * 60)
    print("STEP 5: Storing recommendations in database")
    print("-" * 60)
    success_prep = store_predictions_in_db(conn, forecasts, recommendations)
    success_traffic = store_traffic_recommendations(conn, traffic_recommendations)
    
    # Close the database connection
    conn.close()
    
    # Save forecasts to CSV if requested
    if args.save:
        forecast_file = f"ingredient_forecast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        forecasts['hourly'].to_csv(forecast_file)
        print(f"\n✅ Saved hourly forecasts to {forecast_file}")
        
        # Also save the daily recommendations
        if 'daily_needs' in recommendations:
            daily_file = f"daily_ingredient_needs_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            recommendations['daily_needs'].to_csv(daily_file)
            print(f"✅ Saved daily recommendations to {daily_file}")
            
        # Save traffic recommendations
        if traffic_recommendations:
            traffic_file = f"traffic_recommendations_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(traffic_file, 'w') as f:
                for date, recommendation in traffic_recommendations.items():
                    f.write(f"{date.strftime('%Y-%m-%d')} ({date.strftime('%A')}):\n")
                    f.write(f"{recommendation}\n\n")
            print(f"✅ Saved traffic recommendations to {traffic_file}")
        
        # Save model diagnostic information
        model_diag = f"model_diagnostics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(model_diag, 'w') as f:
            f.write("SARIMAX MODEL DIAGNOSTICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Historical data range: {ingredients_data['hourly'].index.min()} to {ingredients_data['hourly'].index.max()}\n")
            f.write(f"Forecast period: {forecasts['hourly'].index.min()} to {forecasts['hourly'].index.max()}\n\n")
            
            # Add ingredient-specific information
            f.write("TOP INGREDIENTS BY USAGE:\n")
            for ing, total in ingredients_data['hourly'].sum().nlargest(10).items():
                f.write(f"  - {ing}: {total:.2f} units\n")
                
            f.write("\nFORECAST SUMMARY STATISTICS:\n")
            for ing, total in forecasts['hourly'].sum().nlargest(10).items():
                f.write(f"  - {ing}: {total:.2f} units\n")
        
        print(f"✅ Saved model diagnostics to {model_diag}")
    
    # Final summary
    print("\n" + "=" * 60)
    print(f"✅ Prediction complete. Used SARIMAX models for key ingredients.")
    print(f"🔮 Forecast period: {forecasts['hourly'].index.min().strftime('%Y-%m-%d %H:%M')} to {forecasts['hourly'].index.max().strftime('%Y-%m-%d %H:%M')}")
    if success_prep:
        print("✅ Future prep recommendations successfully stored in database.")
    if success_traffic:
        print("✅ Traffic recommendations successfully stored in database.")
    print("=" * 60)