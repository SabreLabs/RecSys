import datetime
import numpy as np
import pandas as pd
from sklearn import metrics, neighbors

def convertCategory(x):
	if x == "S":
		return 13
	return x

def convertTime(x):
	time = datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%S.%fZ")
	epoch = datetime.datetime(1970,1,1)
	return (time-epoch).total_seconds()

print "Loading training data..."
buy_headers = ["session","time","item","price","quantity"]
click_headers = ["session","time","item","category"]
buys = pd.read_csv("data/yoochoose-buys.dat", names=buy_headers)
clicks = pd.read_csv("data/yoochoose-clicks.dat", nrows=100000, dtype=float, names=click_headers, converters={"category": convertCategory, "time": convertTime})

print "Formatting training data..."
sessionFreq = clicks["session"].value_counts()
clicks["sessionFreq"] = clicks["session"].map(lambda x: sessionFreq[x])
clicks["didBuy"] = clicks["session"].isin(buys["session"]).astype(int)

X = clicks.ix[:, 0:5].values
y = clicks.ix[:, 5].values

print "Fitting data..."
alg = neighbors.KNeighborsClassifier()
alg.fit(X,y)

print "Calculating training results..."
print metrics.accuracy_score(y, alg.predict(X))

print "Loading testing data..."
test_buys = pd.read_csv("data/solution.dat", sep=';', names=buy_headers)
test_clicks = pd.read_csv("data/yoochoose-test.dat", dtype=float, names=click_headers, converters={"category": convertCategory, "time": convertTime})

print "Formatting testing data..."
sessionFreq = test_clicks["session"].value_counts()
test_clicks["sessionFreq"] = test_clicks["session"].map(lambda x: sessionFreq[x])
test_clicks["didBuy"] = test_clicks["session"].isin(test_buys["session"]).astype(int)

t_X = test_clicks.ix[:, 0:5].values
t_y = test_clicks.ix[:, 5].values

print "Calculating testing results..."
print metrics.accuracy_score(t_y, np.zeros(len(t_y)))

