import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataPreprocessor:

    def __init__(self, data):
        self.data = data

    def _fill_category_channel_type(self):
        mapping_category_channel_type = self.data.groupby('category')['channel_type'].apply(lambda x: x.mode()[0]).to_dict()
        mapping_channel_type_category = self.data.groupby('channel_type')['category'].apply(lambda x: x.mode()[0]).to_dict()

        # Fill missing channel_type based on category
        mask = self.data['channel_type'].isnull()
        self.data.loc[mask, 'channel_type'] = self.data.loc[mask, 'category'].map(mapping_category_channel_type)

        # Fill missing category based on channel_type
        mask = self.data['category'].isnull()
        self.data.loc[mask, 'category'] = self.data.loc[mask, 'channel_type'].map(mapping_channel_type_category)

    def _fill_country_rank(self):
        self.data['country_rank'] = self.data.groupby('country')['video_views_rank'].transform(lambda x: x.interpolate(method='linear'))

    def _fill_channel_type_rank(self):
        self.data['channel_type_rank'] = self.data.groupby('channel_type')['video_views_rank'].transform(lambda x: x.interpolate(method='linear'))

    def _fill_subscribers_last_30_days(self):
        mask = (self.data['subscribers_for_last_30_days'].isnull()) & (self.data['video_views_rank'] > self.data['video_views_rank'].quantile(0.25))
        self.data.loc[mask, 'subscribers_for_last_30_days'] = 0

        # For the rest of the missing values, use the mean of similar channels
        fill_value = self.data['subscribers_for_last_30_days'].mean()
        self.data['subscribers_for_last_30_days'].fillna(fill_value, inplace=True)

    def _encode_categorical_columns(self):
        label_encoders = {}
        for column in ['category', 'channel_type', 'abbreviation']:
            le = LabelEncoder()
            self.data[column] = le.fit_transform(self.data[column].astype(str))
            label_encoders[column] = le

    def _remove_null_values(self):
        self.data.dropna(inplace=True)

    def preprocess(self):
        self._fill_category_channel_type()
        self._fill_country_rank()
        self._fill_channel_type_rank()
        self._fill_subscribers_last_30_days()
        self._encode_categorical_columns()
        self._remove_null_values()
        return self.data
