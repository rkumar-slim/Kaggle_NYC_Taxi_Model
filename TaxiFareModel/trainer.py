from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
from TaxiFareModel.data import get_data,clean_data
from sklearn.model_selection import train_test_split


class Trainer():
    def __init__(self,
                 X,
                 y,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude",
                 pdt='pickup_datetime',
                 typedist='harversine',
                 pdist=0):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon
        self.pdt = pdt
        self.typedist = typedist
        self.pdist = pdist


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        # create distance pipeline
        DistPipeline = Pipeline([('dist_trans',
                                  DistanceTransformer(self.typedist,
                                                      self.pdist)),
                                 ('stdscaler', StandardScaler())])
        # create time pipeline
        TimePipeline = Pipeline([('time_enc', TimeFeaturesEncoder(self.pdt)),
                                 ('ohe',
                                  OneHotEncoder(handle_unknown='ignore'))])
        # create preprocessing pipeline
        pipeline = ColumnTransformer([('distance', DistPipeline, [
            self.start_lat, self.start_lon, self.end_lat, self.end_lon
        ]), ('time', TimePipeline, [self.pdt])],
                                          remainder="drop")
        self.pipeline = Pipeline([('preproc', pipeline),
                                  ('linear_model', LinearRegression())])


    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        print(self.pipeline.score)


    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        # compute y_pred on the test set
        y_pred = self.pipeline.predict(X_test)
        model_evaluate_emse = compute_rmse(y_pred, y_test)
        return model_evaluate_emse


if __name__ == "__main__":
    # get data
    df = get_data(nrows=10_000)
    # clean data
    df = clean_data(df, test=False)
    # set X and y
    y = df.pop("fare_amount")
    X = df

    # hold out
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    # train
    train_obj = Trainer(X_train, y_train,typedist='minko',pdist=2)
    train_obj.run()

    # evaluate
    mrmse = train_obj.evaluate(X_test, y_test)
    print(f"RMSE value on test data set is {mrmse}")
