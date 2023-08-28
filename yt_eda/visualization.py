import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

class PreprocessVisualizer:

    def __init__(self, data):
        self.data = data

    def plot_numeric_distribution(self):
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.data[col], kde=True)
            plt.title(f"Distribution of {col}")
            plt.show()

    def plot_categorical_distribution(self):
        cat_cols = self.data.select_dtypes(include=['object']).columns
        for col in cat_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(data=self.data, x=col)
            plt.title(f"Distribution of {col}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_channel_category_counts(self, group_first, group_second):
        grouped = self.data.groupby([group_first, group_second]).size().unstack()
        grouped.plot(kind='bar', figsize=(15, 7))
        plt.title(f"Distribution of {group_second} by {group_first}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_heatmap(self):
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.show()


class PostprocessVisualizer:

    def __init__(self, data):
        self.data = data

    def plot_heatmap(self):
        plt.figure(figsize=(12, 12))
        sns.heatmap(self.data.corr(), annot=True, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.show()

    def plot_correlation_analysis(self, target):
        correlations = self.data.corr()[target].sort_values(ascending=False)
        plt.figure(figsize=(15, 6))
        sns.barplot(correlations.index, correlations.values)
        plt.title(f"Correlation with {target}")
        plt.xticks(rotation=45)
        plt.show()

    def plot_scatterplots(self, target):
        features = [col for col in self.data.columns if col != target]
        for feature in features:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=self.data, x=feature, y=target)
            plt.title(f"{feature} vs. {target}")
            plt.show()

    def plot_feature_importance(self, target):
        X = self.data.drop(target, axis=1)
        y = self.data[target]
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        importances = model.feature_importances_
        sorted_idx = importances.argsort()
        plt.figure(figsize=(15, 12))
        plt.barh(X.columns[sorted_idx], importances[sorted_idx])
        plt.xlabel("Feature Importance")
        plt.show()


# Usage:
# pre_viz = PreprocessVisualizer(data)
# post_viz = PostprocessVisualizer(processed_data)
