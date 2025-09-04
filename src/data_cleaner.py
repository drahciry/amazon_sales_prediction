import pandas as pd

def clean_amazon_data(df_raw: pd.DataFrame, top_n_brands: int=25) ->pd.DataFrame:
    """
    This function clean dataset to improve the data analysis
    and model train. Returns a new DataFrame refactored.
    
    Args:
        df_raw (pd.DataFrame): The raw dataset that will be cleaned.
        top_n_brands (int, optional): The quantify of brands that will be
        visible. Defaults to 25.

    Returns:
        DataFrame: DataFrame with cleaned dataset.
    """
    df_cleaned = df_raw.copy()
    
    # Calculate the median of 'price' column
    median_price = df_cleaned["price"].median()

    # Fill missing values in 'price' with the median
    df_cleaned.fillna({"price": median_price}, inplace=True)

    # Do the same to 'rating' column
    median_rating = df_cleaned["rating"].median()
    df_cleaned.fillna({"rating": median_rating}, inplace=True)

    # Deletes all rows with null on any column
    df_cleaned.dropna(inplace=True)

    # Deletes all duplicated rows
    df_cleaned.drop_duplicates(inplace=True)

    # Deletes columns with link
    df_cleaned.drop(columns=['image_url', 'product_url'], inplace=True)

    # Captures the 50 largests brands
    top_brands = df_cleaned['brand'].value_counts().nlargest(25).index

    # Modify the brands that are not in top_brands
    df_cleaned['brand'] = df_cleaned['brand'].where(df_cleaned['brand'].isin(top_brands), 'Other')

    # Sort values by title
    df_cleaned.sort_values('title', inplace=True)