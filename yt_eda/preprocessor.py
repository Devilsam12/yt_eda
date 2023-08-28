"""
This module provides functionalities for preprocessing data. This includes 
handling missing values, encoding categorical variables, and converting datetime 
data into a more usable format.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class DataPreprocessor:
    """
    Preprocessor class encapsulates methods designed for various preprocessing 
    tasks on a dataset.
    """

    def __init__(self, data):
        """
        Initialize the Preprocessor class.

        Parameters:
        - data: pd.DataFrame
            The input dataset that requires preprocessing.
        """
        self.data = data

    def _fill_category_channel_type(self):
        """
        Fill missing values in 'category' and 'channel_type' columns using a 
        mode-based approach.
        """

        mapping_category_channel_type = self.data.groupby('category')['channel_type'].apply(lambda x: x.mode()[0]).to_dict()
        mapping_channel_type_category = self.data.groupby('channel_type')['category'].apply(lambda x: x.mode()[0]).to_dict()

        # Fill missing channel_type based on category
        mask = self.data['channel_type'].isnull()
        self.data.loc[mask, 'channel_type'] = self.data.loc[mask, 'category'].map(mapping_category_channel_type)

        # Fill missing category based on channel_type
        mask = self.data['category'].isnull()
        self.data.loc[mask, 'category'] = self.data.loc[mask, 'channel_type'].map(mapping_channel_type_category)

    def _fill_country_rank(self):
        """
        Fill missing values in 'country_rank' based on the ordering of 'video_views_rank' 
        within the same country.
        """
        self.data['country_rank'] = self.data.groupby('Country')['video_views_rank'].transform(lambda x: x.interpolate(method='linear'))

    def _fill_channel_type_rank(self):
        """
        Fill missing values in 'channel_type_rank' based on the ordering of 'video_views_rank' 
        within the same channel_type.
        """
        self.data['channel_type_rank'] = self.data.groupby('channel_type')['video_views_rank'].transform(lambda x: x.interpolate(method='linear'))

    def _fill_subscribers_last_30_days(self):
        """
        Fill missing values in 'subscribers_for_last_30_days' using a defined strategy 
        based on 'video_views_rank'.
        """
        mask = (self.data['subscribers_for_last_30_days'].isnull()) & (self.data['video_views_rank'] > self.data['video_views_rank'].quantile(0.25))
        self.data.loc[mask, 'subscribers_for_last_30_days'] = 0

        # For the rest of the missing values, use the mean of similar channels
        fill_value = self.data['subscribers_for_last_30_days'].mean()
        self.data['subscribers_for_last_30_days'].fillna(fill_value, inplace=True)

    def _encode_categorical_columns(self):
        """
        Convert categorical columns like 'category', 'channel_type', and 'abbreviation' 
        into numerical representations using label encoding.
        """
        label_encoders = {}
        for column in ['category', 'channel_type', 'Abbreviation']:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column].astype(str))
            label_encoders[column] = le
    
    def _handle_datetime_and_age(self):
        """
        Convert 'created_month', 'created_year', and 'created_date' into a single datetime 
        column, and subsequently calculate the age of each entry in months.
        """
        month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                      'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
        
        # Convert to numeric
        self.data['created_month'] = self.data['created_month'].map(month_dict).astype(int)
        self.data['created_year'] = self.data['created_year'].astype(int)
        self.data['created_date'] = self.data['created_date'].astype(int)

        # Create datetime column
        self.data['created_datetime'] = pd.to_datetime(self.data[['created_year', 'created_month', 'created_date']])
        
        # Calculate age in months
        current_date = datetime.now()
        diff = (current_date - self.data['created_datetime'])
        self.data['age_in_months'] = diff.dt.days // 30
        
        # Drop unnecessary columns
        self.data.drop(columns=['created_year', 'created_month', 'created_date', 'created_datetime'], inplace=True)


    def _remove_null_values(self):
        """
        Remove rows from the dataset where any of the elements is nan.
        """
        self.data.dropna(inplace=True)

    def preprocess(self):
        """
        Execute a series of preprocessing tasks on the dataset.

        This method carries out the following operations in order:
        1. Fills missing values in the 'category' and 'channel_type' columns.
        2. Encodes categorical columns such as 'category', 'channel_type', and 'abbreviation' into numerical labels.
        3. Handles missing values in 'country_rank' and 'channel_type_rank' based on 'video_views_rank'.
        4. Fills missing values in 'subscribers_for_last_30_days' based on a specific strategy.
        5. Removes rows with any NaN values.
        
        Returns:
        - pd.DataFrame: The preprocessed dataset.
        """
        self._fill_category_channel_type()
        self._fill_country_rank()
        self._fill_channel_type_rank()
        self._fill_subscribers_last_30_days()
        self._encode_categorical_columns()
        self._remove_null_values()
        return self.data
