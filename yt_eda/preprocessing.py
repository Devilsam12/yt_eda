import pandas as pd
from sklearn.preprocessing import LabelEncoder

def fill_category_channel_type(data):
    channel_to_category = data.groupby('channel_type')['category'].apply(lambda x: x.mode()[0]).to_dict()
    category_to_channel = data.groupby('category')['channel_type'].apply(lambda x: x.mode()[0]).to_dict()

    data['channel_type'] = data.apply(
        lambda row: channel_to_category[row['category']] if pd.isnull(row['channel_type']) else row['channel_type'],
        axis=1
    )

    data['category'] = data.apply(
        lambda row: category_to_channel[row['channel_type']] if pd.isnull(row['category']) else row['category'],
        axis=1
    )

    return data

def fill_country_rank(data):
    country_groups = data.groupby('country')
    for _, group in country_groups:
        ordered_idxs = group['video_views_rank'].argsort().values
        filled_values = group.loc[ordered_idxs, 'country_rank'].interpolate().bfill().ffill().values
        data.loc[group.index, 'country_rank'] = filled_values

    return data

def fill_channel_type_rank(data):
    channel_type_groups = data.groupby('channel_type')
    for _, group in channel_type_groups:
        ordered_idxs = group['video_views_rank'].argsort().values
        filled_values = group.loc[ordered_idxs, 'channel_type_rank'].interpolate().bfill().ffill().values
        data.loc[group.index, 'channel_type_rank'] = filled_values

    return data

def fill_subscribers_last_30_days(data):
    mask = (data['video_views_rank'] > data['video_views_rank'].quantile(0.25)) & data['subscribers_for_last_30_days'].isna()
    data.loc[mask, 'subscribers_for_last_30_days'] = 0

    median_subs = data['subscribers_for_last_30_days'].median()
    data['subscribers_for_last_30_days'].fillna(median_subs, inplace=True)

    return data

def encode_categorical_columns(data):
    label_encoders = {}
    for col in ['category', 'channel_type', 'abbreviation']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le

    return data, label_encoders

def remove_null_values(data):
    return data.dropna()

def preprocess_data(data):
    data = fill_category_channel_type(data)
    data = fill_country_rank(data)
    data = fill_channel_type_rank(data)
    data = fill_subscribers_last_30_days(data)
    data, _ = encode_categorical_columns(data)
    data = remove_null_values(data)

    return data

