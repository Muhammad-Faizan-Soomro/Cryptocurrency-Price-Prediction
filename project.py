import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import yfinance as yf
import datetime as dt
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle
import calendar
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


# import tensorflow.compat.v1 as tf


st.header("Crypto Price Prediction")
st.info(
    "This Will Give Analysis on the present and last ten year prices of crypto coins"
)

# Input for coin name
coin_name = st.text_input("Enter the cryptocurrency symbol (e.g., BTC, ETH)", "BTC")


def remove(x):
    """
    This function will strip the data column of the dataframe.
    """
    x = str(x)
    res = x.split(" ")[0]
    return res


data = yf.download(coin_name + "-USD", period="max")

st.subheader("DataSet")
st.write(data)

data.index = pd.to_datetime(data.index, format="%Y-%m-%d")
data.index = data.index.to_series().apply(
    lambda x: remove(x)
)  # applying preprocessing function

Eda, DataPreprocessing, Models, Compare = st.tabs(
    ["Exploratory Data Analysis", "Data PreProcessing", "Models", "Final Comparison"]
)

with Eda:
    st.subheader("Shape")
    shape = data.shape
    st.write(shape)

    st.subheader("Head")
    head = data.head()
    st.write(head)

    st.subheader("Tail")
    tail = data.tail()
    st.write(tail)

    st.subheader("Info")
    info = data.info
    st.write(info)

    st.subheader("Describe")
    describe = data.describe()
    st.write(describe)

    st.subheader("Index")
    index = data.index
    st.write(index)

    st.subheader("Null Values")
    st.write("Null Values:", data.isnull().values.sum())
    st.write("NA values:", data.isnull().values.any())
    st.info("This Shows that there are no null values")

    st.title("Year Wise Distribution Of The DataSet")

    new_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Define functions specific to EDA tab
    def yearly_analysis(start_date, end_date):
        year = data.loc[start_date:end_date]
        year.index = pd.to_datetime(year.index, format="%Y-%m-%d")
        return year

    def yearly_chart(yearlyDataset):
        names = cycle([coin_name + " Close Price", coin_name])

        fig = px.line(
            yearlyDataset,
            x=yearlyDataset.index,
            y="Close",
            labels={"Date": "Date", "value": coin_name + " value"},
        )
        fig.update_layout(
            title_text=coin_name + " analysis chart",
            font_size=15,
            font_color="black",
            legend_title_text=coin_name + " Parameters",
        )
        fig.for_each_trace(lambda t: t.update(name=next(names)))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

        st.plotly_chart(fig)

        figure = go.Figure(
            data=[
                go.Candlestick(
                    x=yearlyDataset.index,
                    low=yearlyDataset["Low"],
                    high=yearlyDataset["High"],
                    close=yearlyDataset["Close"],
                    open=yearlyDataset["Open"],
                    increasing_line_color="green",
                    decreasing_line_color="red",
                )
            ]
        )
        st.plotly_chart(figure)

    def every_year_monthwise_analysis(yearlyDataset):
        monthwise = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            ["Open", "Close"]
        ].mean()
        monthwise = monthwise.reindex(new_order, axis=0)
        return monthwise

    def monthly_open_close_chart(monthlyDataset):
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=monthlyDataset.index,
                y=monthlyDataset["Open"],
                name=coin_name + " Open Price",
                marker_color="crimson",
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthlyDataset.index,
                y=monthlyDataset["Close"],
                name=coin_name + " Close Price",
                marker_color="lightsalmon",
            )
        )

        fig.update_layout(
            barmode="group",
            xaxis_tickangle=-45,
            title="Monthwise comparison between " + coin_name + " open and close price",
        )
        st.plotly_chart(fig)

    def monthly_high_low_chart(yearlyDataset):
        yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))["Low"].min()
        monthwise_high = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            "High"
        ].max()
        monthwise_high = monthwise_high.reindex(new_order, axis=0)

        monthwise_low = yearlyDataset.groupby(yearlyDataset.index.strftime("%B"))[
            "Low"
        ].min()
        monthwise_low = monthwise_low.reindex(new_order, axis=0)

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=monthwise_high.index,
                y=monthwise_high,
                name=coin_name + " High Price",
                marker_color="rgb(0, 153, 204)",
            )
        )
        fig.add_trace(
            go.Bar(
                x=monthwise_low.index,
                y=monthwise_low,
                name=coin_name + " Low Price",
                marker_color="rgb(255, 128, 0)",
            )
        )

        fig.update_layout(
            barmode="group", title="Monthwise High and Low " + coin_name + " price"
        )
        st.plotly_chart(fig)

    # Input for start and end dates
    start_date = st.date_input("Start date")
    end_date = st.date_input("End date")

    if start_date and end_date:
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

    if st.button("Analyze"):
        year_2014 = yearly_analysis(start_date, end_date)
        monthly_2014 = every_year_monthwise_analysis(year_2014)

        st.subheader(f"Yearly Chart for {coin_name}")
        yearly_chart(year_2014)

        st.subheader(f"Monthly Open and Close Chart for {coin_name}")
        monthly_open_close_chart(monthly_2014)

        st.subheader(f"Monthly High and Low Chart for {coin_name}")
        monthly_high_low_chart(year_2014)

        st.subheader(f"Heatmap of Monthly Volume for {coin_name}")
        data_copy = data.copy()
        data_copy.index = pd.to_datetime(data_copy.index, format="%Y-%m-%d")
        data_copy["Month"] = data_copy.index.month
        data_copy["Year"] = data_copy.index.year
        grouped_data = (
            data_copy.groupby(["Year", "Month"])["Volume"].sum().reset_index()
        )
        grouped_data["Month"] = grouped_data["Month"].apply(
            lambda x: calendar.month_name[x]
        )
        x = grouped_data.pivot_table(
            index="Year", columns="Month", values="Volume", aggfunc="sum"
        )
        x.columns = pd.Categorical(x.columns, categories=new_order, ordered=True)
        x = x.sort_index(axis=1)

        # Plotting heatmap with Plotly
        fig_heatmap = go.Figure(
            data=go.Heatmap(z=x.values, x=x.columns, y=x.index, colorscale="Viridis")
        )
        fig_heatmap.update_layout(
            title=f"Heatmap of Monthly Volume for {coin_name}",
            xaxis_title="Month",
            yaxis_title="Year",
        )
        st.plotly_chart(fig_heatmap)

        DSR = data["Close"].pct_change(1)
        st.write(DSR)
        st.write(DSR.describe())
        st.info(
            f"if we invest in {coin_name}, we can approx get 0.2% return daily  investment has gome to 37% daily loss sometimes investment has gone to 25% daily profit sometimes ( max price increased in a day is 25% "
        )


with DataPreprocessing:
    st.title("Feature Engineering")

    def SMA(dataset, period=30, column="Close"):
        return dataset[column].rolling(window=period).mean()

    def EMA(dataset, period=20, column="Close"):
        return dataset[column].ewm(span=period, adjust=False).mean()

    def MACD(data, period_long=26, period_short=12, period_signal=9, column="Close"):
        ShortEMA = EMA(data, period_short, column=column)
        LongEMA = EMA(data, period_long, column=column)
        data["MACD"] = ShortEMA - LongEMA
        data["Signal_Line"] = EMA(data, period_signal, column="MACD")
        return data

    def RSI(data, period=14, column="Close"):
        delta = data[column].diff(1)
        delta = delta[1:]
        up = delta.copy()
        down = delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0
        data["up"] = up
        data["down"] = down
        AVG_Gain = SMA(data, period, column="up")
        AVG_Loss = abs((SMA(data, period, column="down")))
        RS = AVG_Gain / AVG_Loss
        RSI = 100.0 - (100.0 / (1.0 + RS))
        data["RSI"] = RSI
        return data

    MACD(data)
    RSI(data)
    data["SMA"] = SMA(data)
    data["EMA"] = EMA(data)

    future_days = st.text_input("enter the number of future predictions days")
    if future_days:
        try:
            # Convert input to integer
            future_days = int(future_days)

            # Create a new column for future predictions
            data[str(future_days) + "_Days_Price_Pred"] = data[["Close"]].shift(
                -future_days
            )

            # Select relevant columns to display
            future_data = data[["Close", str(future_days) + "_Days_Price_Pred"]]

            # Display the DataFrame
            st.write(future_data)
        except ValueError:
            st.error("Invalid input. Please enter a valid number of days.")

    st.title("Scaling The Data")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)

    st.write(scaled_df)


with Models:

    WPP, PP = st.tabs(["Without Python Package", "With Python Package"])

    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        return mse, rmse, r2, mae

    with PP:

        # future_days = st.number_input("Enter the days for future prediction")
        if future_days:
            st.header("Using All  Column")
            columns_to_extract = [
                "Close",
                "MACD",
                "RSI",
                "SMA",
                "EMA",
            ]  # specify the columns you want to extract
            X = np.array(scaled_df[columns_to_extract])
            X = X[30 : scaled_df.shape[0] - future_days]
            y = np.array(scaled_df[str(future_days) + "_Days_Price_Pred"])
            y = y[30:-future_days]

            tscv = TimeSeriesSplit(n_splits=5)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

            st.write(f"X_train shape: {X_train.shape}")
            st.write(f"X_test shape: {X_test.shape}")
            st.write(f"y_train shape: {y_train.shape}")
            st.write(f"y_test shape: {y_test.shape}")

            st.subheader("SVM MODEL")
            svr_python = SVR(kernel="rbf", C=1e3, gamma=0.00001)
            svr_python.fit(X_train, y_train)
            svr_python_confidence = svr_python.score(X_test, y_test)
            st.write(svr_python_confidence)

            (
                svr_python_mse_value,
                svr_python_rmse_value,
                svr_python_r2_value,
                svr_python_mae_value,
            ) = evaluate_model(svr_python, X_test, y_test)
            st.write(f"SVR (with package) MSE: {svr_python_mse_value}")
            st.write(f"SVR (with package) RMSE: {svr_python_rmse_value}")
            st.write(f"SVR (with package) R-squared: {svr_python_r2_value}")
            st.write(f"SVR (with package) MAE: {svr_python_mae_value}")

            st.subheader("Linear Regression")
            lr_python = LinearRegression()
            lr_python.fit(X_train, y_train)
            lr_python_confidence = lr_python.score(X_test, y_test)
            st.write(lr_python_confidence)

            (
                lr_python_mse_value,
                lr_python_rmse_value,
                lr_python_r2_value,
                lr_python_mae_value,
            ) = evaluate_model(lr_python, X_test, y_test)
            st.write(f"LR (with package) MSE: {lr_python_mse_value}")
            st.write(f"LR (with package) RMSE: {lr_python_rmse_value}")
            st.write(f"LR (with package) R-squared: {lr_python_r2_value}")
            st.write(f"LR (with package) MAE: {lr_python_mae_value}")

            st.subheader("Random Forest")
            rf_python = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_python.fit(X_train, y_train)
            rf_python_confidence = rf_python.score(X_test, y_test)
            st.write(rf_python_confidence)

            (
                rf_python_mse_value,
                rf_python_rmse_value,
                rf_python_r2_value,
                rf_python_mae_value,
            ) = evaluate_model(rf_python, X_test, y_test)
            st.write(f"RF (with package) MSE: {rf_python_mse_value}")
            st.write(f"RF (with package) RMSE: {rf_python_rmse_value}")
            st.write(f"RF (with package) R-squared: {rf_python_r2_value}")
            st.write(f"RF (with package) MAE: {rf_python_mae_value}")

            def evaluate_model(model, X_test, y_test):
                y_pred = model.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                return {"MSE": mse, "RMSE": rmse, "R-squared": r2, "MAE": mae}

            rf_python_metrics = evaluate_model(rf_python, X_test, y_test)
            st.write("Random Forest (with package) Metrics:")
            st.write(rf_python_metrics)

            lr_python_metrics = evaluate_model(lr_python, X_test, y_test)
            st.write("\nLinear Regression (with package) Metrics:")
            st.write(lr_python_metrics)

            svr_python_metrics = evaluate_model(svr_python, X_test, y_test)
            st.write("\nSupport Vector Machine (with package) Metrics:")
            st.write(svr_python_metrics)

            st.header("Calculating Predictions")
            svm_python_predictions = svr_python.predict(X_test)
            lr_python_predictions = lr_python.predict(X_test)
            rf_python_predictions = rf_python.predict(X_test)
            st.subheader("random forest")
            st.write(rf_python_predictions)
            st.subheader("linear regression ")
            st.write(lr_python_predictions)
            st.subheader("svm")
            st.write(svm_python_predictions)

            st.header("Plotting Predictions Vs Actual")
            # Plotting the results
            plt.figure(figsize=(14, 7))

            # # Linear Regression
            plt.subplot(3, 1, 1)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(lr_python_predictions, label="Linear Regression", color="orange")
            plt.title("Linear Regression ( with package ) vs Actual")
            plt.legend()

            # Random Forest
            plt.subplot(3, 1, 2)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(rf_python_predictions, label="Random Forest", color="green")
            plt.title("Random Forest ( with package ) vs Actual")
            plt.legend()

            # SVM
            plt.subplot(3, 1, 3)
            plt.plot(y_test, label="Actual", color="blue")
            plt.plot(svm_python_predictions, label="SVR", color="red")
            plt.title("SVR ( with package ) vs Actual")
            plt.legend()

            plt.tight_layout()
            plt.show()
            st.pyplot(plt)

    with WPP:
        if future_days:

            st.title("Random Forest Model")

            def bootstrap_sample(X, y):
                n_samples = X.shape[0]
                idxs = np.random.choice(n_samples, n_samples, replace=True)
                return X[idxs], y[idxs]

            def mean_squared_error(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)

            class Node:
                def __init__(
                    self,
                    feature=None,
                    threshold=None,
                    left=None,
                    right=None,
                    *,
                    value=None,
                ):
                    self.feature = feature
                    self.threshold = threshold
                    self.left = left
                    self.right = right
                    self.value = value

                def is_leaf_node(self):
                    return self.value is not None

            class DecisionTreeRegressor:
                def __init__(self, max_depth=100, min_samples_split=2):
                    self.max_depth = max_depth
                    self.min_samples_split = min_samples_split
                    self.root = None

                def fit(self, X, y):
                    self.root = self._grow_tree(X, y)

                def _grow_tree(self, X, y, depth=0):
                    n_samples, n_features = X.shape
                    if depth >= self.max_depth or n_samples < self.min_samples_split:
                        leaf_value = self._calculate_leaf_value(y)
                        return Node(value=leaf_value)

                    feat_idxs = np.random.choice(n_features, n_features, replace=True)
                    best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

                    if best_thresh is None:
                        leaf_value = self._calculate_leaf_value(y)
                        return Node(value=leaf_value)

                    left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
                    left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
                    right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
                    return Node(best_feat, best_thresh, left, right)

                def _best_criteria(self, X, y, feat_idxs):
                    best_mse = float("inf")
                    split_idx, split_thresh = None, None
                    for feat_idx in feat_idxs:
                        X_column = X[:, feat_idx]
                        thresholds = np.unique(X_column)
                        for threshold in thresholds:
                            mse = self._calculate_mse(y, X_column, threshold)
                            if mse < best_mse:
                                best_mse = mse
                                split_idx = feat_idx
                                split_thresh = threshold
                    return split_idx, split_thresh

                def _calculate_mse(self, y, X_column, split_thresh):
                    left_idxs, right_idxs = self._split(X_column, split_thresh)
                    if len(left_idxs) == 0 or len(right_idxs) == 0:
                        return float("inf")

                    y_left, y_right = y[left_idxs], y[right_idxs]
                    mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
                    mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
                    n, n_left, n_right = len(y), len(y_left), len(y_right)
                    mse = (n_left / n) * mse_left + (n_right / n) * mse_right
                    return mse

                def _split(self, X_column, split_thresh):
                    left_idxs = np.argwhere(X_column <= split_thresh).flatten()
                    right_idxs = np.argwhere(X_column > split_thresh).flatten()
                    return left_idxs, right_idxs

                def _calculate_leaf_value(self, y):
                    return np.mean(y)

                def predict(self, X):
                    return np.array([self._traverse_tree(x, self.root) for x in X])

                def _traverse_tree(self, x, node):
                    if node.is_leaf_node():
                        return node.value
                    if x[node.feature] <= node.threshold:
                        return self._traverse_tree(x, node.left)
                    return self._traverse_tree(x, node.right)

            class RandomForestRegressorScratch:
                def __init__(self, n_trees=100, max_depth=100, min_samples_split=2):
                    self.n_trees = n_trees
                    self.max_depth = max_depth
                    self.min_samples_split = min_samples_split
                    self.trees = []

                def fit(self, X, y):
                    self.trees = []
                    for _ in range(self.n_trees):
                        tree = DecisionTreeRegressor(
                            max_depth=self.max_depth,
                            min_samples_split=self.min_samples_split,
                        )
                        X_sample, y_sample = bootstrap_sample(X, y)
                        tree.fit(X_sample, y_sample)
                        self.trees.append(tree)

                def predict(self, X):
                    tree_preds = np.array([tree.predict(X) for tree in self.trees])
                    return np.mean(tree_preds, axis=0)

                def score(self, X, y):
                    y_pred = self.predict(X)
                    return mean_squared_error(y, y_pred)

                # Example usage (ensure X_train, y_train, X_test, y_test are defined)
                # X_train, y_train, X_test, y_test = <your data loading code>

                # Ensure X_train and y_train are numpy arrays
                # X_train = np.array(X_train)
                # y_train = np.array(y_train)

            rf_scratch = RandomForestRegressorScratch(n_trees=3, max_depth=10)
            rf_scratch.fit(X_train, y_train)

            (
                rf_scratch_mse_value,
                rf_scratch_rmse_value,
                rf_scratch_r2_value,
                rf_scratch_mae_value,
            ) = evaluate_model(rf_scratch, X_test, y_test)
            st.write(f"RF (w/o package) MSE: {rf_scratch_mse_value}")
            st.write(f"RF (w/o package) RMSE: {rf_scratch_rmse_value}")
            st.write(f"RF (w/o package) R-squared: {rf_scratch_r2_value}")
            st.write(f"RF (w/o package) MAE: {rf_scratch_mae_value}")

            st.title("LInear Regression Model")

            class LinearRegressionScratch:
                def __init__(self, lr=0.001, n_iters=1000):
                    self.lr = lr
                    self.n_iters = n_iters
                    self.weights = None
                    self.bias = None

                def fit(self, X, y):
                    n_samples, n_features = X.shape
                    self.weights = np.zeros(n_features)
                    self.bias = 0

                    for _ in range(self.n_iters):
                        y_pred = np.dot(X, self.weights) + self.bias

                        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                        db = (1 / n_samples) * np.sum(y_pred - y)

                        self.weights = self.weights - self.lr * dw
                        self.bias = self.bias - self.lr * db

                def predict(self, X):
                    y_pred = np.dot(X, self.weights) + self.bias
                    return y_pred

            lr_scratch = LinearRegressionScratch(lr=0.01)
            lr_scratch.fit(X_train, y_train)
            (
                lr_scratch_mse_value,
                lr_scratch_rmse_value,
                lr_scratch_r2_value,
                lr_scratch_mae_value,
            ) = evaluate_model(lr_scratch, X_test, y_test)
            st.write(f"LR (w/o package) MSE: {lr_scratch_mse_value}")
            st.write(f"LR (w/o package) RMSE: {lr_scratch_rmse_value}")
            st.write(f"LR (w/o package) R-squared: {lr_scratch_r2_value}")
            st.write(f"LR (w/o package) MAE: {lr_scratch_mae_value}")

            st.title("SVM Model")

            class SVRscratch(object):
                def __init__(self, epsilon=0.5):
                    self.epsilon = epsilon

                def fit(self, X, y, epochs=100, learning_rate=0.1):
                    self.sess = tf.Session()

                    feature_len = X.shape[-1] if len(X.shape) > 1 else 1

                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)
                    if len(y.shape) == 1:
                        y = y.reshape(-1, 1)

                    self.X = tf.placeholder(dtype=tf.float32, shape=(None, feature_len))
                    self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1))

                    self.W = tf.Variable(tf.random_normal(shape=(feature_len, 1)))
                    self.b = tf.Variable(tf.random_normal(shape=(1,)))

                    self.y_pred = tf.matmul(self.X, self.W) + self.b

                    # self.loss = tf.reduce_mean(tf.square(self.y - self.y_pred))
                    # self.loss = tf.reduce_mean(tf.cond(self.y_pred - self.y < self.epsilon, lambda: 0, lambda: 1))

                    # Second part of following equation, loss is a function of how much the error exceeds a defined value, epsilon
                    # Error lower than epsilon = no penalty.
                    self.loss = tf.norm(self.W) / 2 + tf.reduce_mean(
                        tf.maximum(0.0, tf.abs(self.y_pred - self.y) - self.epsilon)
                    )
                    #         self.loss = tf.reduce_mean(tf.maximum(0., tf.abs(self.y_pred - self.y) - self.epsilon))

                    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
                    opt_op = opt.minimize(self.loss)

                    self.sess.run(tf.global_variables_initializer())

                    for i in range(epochs):
                        loss = self.sess.run(self.loss, {self.X: X, self.y: y})
                        print("{}/{}: loss: {}".format(i + 1, epochs, loss))

                        self.sess.run(opt_op, {self.X: X, self.y: y})

                    return self

                def predict(self, X, y=None):
                    if len(X.shape) == 1:
                        X = X.reshape(-1, 1)

                    y_pred = self.sess.run(self.y_pred, {self.X: X})
                    return y_pred

        svr_scratch = SVRscratch(epsilon=0.01)
        svr_scratch.fit(X_train, y_train)
        (
            svr_scratch_mse_value,
            svr_scratch_rmse_value,
            svr_scratch_r2_value,
            svr_scratch_mae_value,
        ) = evaluate_model(svr_scratch, X_test, y_test)
        st.write(f"SVR (w/o package) MSE: {svr_scratch_mse_value}")
        st.write(f"SVR (w/o package) RMSE: {svr_scratch_rmse_value}")
        st.write(f"SVR (w/o package) R-squared: {svr_scratch_r2_value}")
        st.write(f"SVR (w/o package) MAE: {svr_scratch_mae_value}")

        svm_scratch_predictions = svr_scratch.predict(X_test)
        lr_scratch_predictions = lr_scratch.predict(X_test)
        rf_scratch_predictions = rf_scratch.predict(X_test)

        # Plotting the results
        plt.figure(figsize=(14, 7))

        # Linear Regression
        plt.subplot(3, 1, 1)
        plt.plot(y_test, label="Actual", color="blue")
        plt.plot(lr_scratch_predictions, label="Linear Regression", color="orange")
        plt.title("Linear Regression ( w/o package ) vs Actual")
        plt.legend()

        # Random Forest
        plt.subplot(3, 1, 2)
        plt.plot(y_test, label="Actual", color="blue")
        plt.plot(rf_scratch_predictions, label="Random Forest", color="green")
        plt.title("Random Forest ( w/o package ) vs Actual")
        plt.legend()

        # SVM
        plt.subplot(3, 1, 3)
        plt.plot(y_test, label="Actual", color="blue")
        plt.plot(svm_scratch_predictions, label="SVR", color="red")
        plt.title("SVR ( w/o package ) vs Actual")
        plt.legend()

        plt.tight_layout()
        plt.show()
        st.plot(plt)


with Compare:
    if future_days:
        comparision_data = {
            "MSE": [
                svr_python_mse_value,
                svr_scratch_mse_value,
                "",
                rf_python_mse_value,
                rf_scratch_mse_value,
                "",
                lr_python_mse_value,
                lr_scratch_mse_value,
            ],
            "RMSE": [
                svr_python_rmse_value,
                svr_scratch_rmse_value,
                "",
                rf_python_rmse_value,
                rf_scratch_rmse_value,
                "",
                lr_python_rmse_value,
                lr_scratch_rmse_value,
            ],
            "R^2": [
                svr_python_r2_value,
                svr_scratch_r2_value,
                "",
                rf_python_r2_value,
                rf_scratch_r2_value,
                "",
                lr_python_r2_value,
                lr_scratch_r2_value,
            ],
            "MAE": [
                svr_python_mae_value,
                svr_scratch_mae_value,
                "",
                rf_python_mae_value,
                rf_scratch_mae_value,
                "",
                lr_python_mae_value,
                lr_scratch_mae_value,
            ],
        }

        comparision_df = pd.DataFrame(comparision_data)

    algo = [
        "SVR (with package)",
        "SVR (w/o package)",
        "",
        "RF (with package)",
        "RF (w/o package)",
        "",
        "LR (with package)",
        "LR (w/o package)",
    ]

    comparision_df.index = algo

    st.write(comparision_df)
