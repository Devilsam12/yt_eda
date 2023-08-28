import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

class FeatureEngineer:
    """
    Handles feature engineering tasks for the dataset.
    
    Methods:
    - execute_engineering: Perform all the feature engineering tasks on the data.
    - average_earnings: Create an average earnings column.
    - remove_low_importance_features: Removes features with importance less than a given threshold.
    """

    def __init__(self, data, target):
        """
        Initialize the FeatureEngineering object with data and target column.
        
        Parameters:
        - data (pd.DataFrame): Input dataset for feature engineering.
        - target (str, optional): Target column for the prediction. Default is 'average_yearly_earnings'.
        """
        self.data = data
        self.target = target

    def _find_feature_importance(self):
        X = self.data.drop(self.target, axis=1)
        y = self.data[self.target]
        
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        return model.feature_importances_

    def _remove_low_importance_features(self, threshold=0.004):
        """
        Remove features with importance values less than the specified threshold.

        Based on a fitted Random Forest model, this method identifies and drops columns/features 
        with importance values less than the given threshold.

        Parameters:
        - threshold (float, optional): Importance value threshold. Default is 0.004.

        Returns:
        - pd.DataFrame: Dataset without the low importance features.
        """
        importances = self._find_feature_importance()
        print(importances)
        columns_to_consider = self.data.drop(columns=[self.target])
        columns_to_drop = columns_to_consider.columns[importances < threshold].tolist()
        self.data.drop(columns=columns_to_drop, inplace=True)

    def _remove_selected_columns(self):
        columns_to_remove = ["Youtuber", "Title", "Country"]
        self.data.drop(columns=columns_to_remove, inplace=True, errors='ignore')  # Using errors='ignore' to avoid errors if columns don't exist.

    def _calculate_average_yearly_earnings(self):
        """
        Calculate the average yearly earnings column.
        
        Creates a new column 'average_yearly_earnings' by averaging 'highest_yearly_earnings' and 'lowest_yearly_earnings' columns.
        
        Returns:
        - pd.DataFrame: Dataset with the new 'average_yearly_earnings' column.
        """
        self.data["avg_yearly_earnings"] = (self.data["lowest_yearly_earnings"] + self.data["highest_yearly_earnings"]) / 2

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
        self.data['created_datetime'] = pd.to_datetime(self.data['created_year'].astype(str) + 
                                               '-' + self.data['created_month'].astype(str) + 
                                               '-' + self.data['created_date'].astype(str))

        # Calculate age in months
        current_date = datetime.now()
        diff = (current_date - self.data['created_datetime'])
        self.data['age_in_months'] = diff.dt.days // 30
        
        # Drop unnecessary columns
        self.data.drop(columns=['created_year', 'created_month', 'created_date', 'created_datetime'], inplace=True)


    def engineer_features(self):
        """
        Execute all feature engineering tasks on the dataset.
        1. Computes created date and age of channel
        2. Computes average yearly earnings.
        3. Removes columns with low importance based on a Random Forest model.
        4. Drops specific columns which are not needed for prediction.
        
        Returns:
        - pd.DataFrame: Dataset after all feature engineering tasks.
        """
        self._handle_datetime_and_age()
        self._remove_selected_columns()
        self._calculate_average_yearly_earnings()
        #self._remove_low_importance_features()
        self._drop_earnings_columns()
        return self.data

# Usage:
# engineer = FeatureEngineer(data, 'avg_yearly_earnings')
# engineered_data = engineer.engineer_features()
