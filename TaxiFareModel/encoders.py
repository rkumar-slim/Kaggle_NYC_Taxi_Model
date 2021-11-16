from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized, minkowski_distance_gps

# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """
    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        X_.index = pd.to_datetime(X[self.time_column])
        X_.index = X_.index.tz_convert(self.time_zone_name)
        X_["dow"] = X_.index.weekday
        X_["hour"] = X_.index.hour
        X_["month"] = X_.index.month
        X_["year"] = X_.index.year
        return X_[['dow', 'hour', 'month', 'year']]


class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """
    def __init__(self,
                 typedist,
                 pdist,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.typedist = typedist
        self.pdist = pdist

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X_ = X.copy()
        if self.typedist == 'harversine':
            X_["distance"] = haversine_vectorized(X_,
                                                start_lat=self.start_lat,
                                                start_lon=self.start_lon,
                                                end_lat=self.end_lat,
                                                end_lon=self.end_lon)
        else:
            X_["distance"] = \
                minkowski_distance_gps(X_[self.start_lat], \
                    X_[self.end_lat],\
                    X_[self.start_lon], \
                    X_[self.end_lon], self.pdist)

        return X_[['distance']]
