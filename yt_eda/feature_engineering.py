import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class FeatureEngineer:

    def __init__(self, data, target):
        self.data = data
        self.target = target

    def _find_feature_importance(self):
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        return model.feature_importances_

    def _remove_low_importance_features(self, threshold=0.004):
        importances = self._find_feature_importance()
        columns_to_drop = self.data.columns[importances < threshold].tolist()
        self.data.drop(columns=columns_to_drop, inplace=True)

    def _remove_selected_columns(self):
        columns_to_remove = ["youtuber", "Title", "Country"]
        self.data.drop(columns=columns_to_remove, inplace=True, errors='ignore')  # Using errors='ignore' to avoid errors if columns don't exist.

    def _calculate_average_yearly_earnings(self):
        self.data["average_yearly_earnings"] = (self.data["lowest_yearly_earnings"] + self.data["highest_yearly_earnings"]) / 2

    def _drop_earnings_columns(self):
        columns_to_remove = ["lowest_monthly_earnings", "highest_monthly_earnings", "lowest_yearly_earnings", "highest_yearly_earnings"]
        self.data.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    def _handle_datetime_and_age(self):
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


    def engineer_features(self):
        self._remove_low_importance_features()
        self._remove_selected_columns()
        self._calculate_average_yearly_earnings()
        self._drop_earnings_columns()
        return self.data

# Usage:
# engineer = FeatureEngineer(data, 'avg_yearly_earnings')
# engineered_data = engineer.engineer_features()
