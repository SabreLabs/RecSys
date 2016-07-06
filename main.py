import datetime
import numpy as np
import pandas as pd
from sklearn import linear_model

def calculateAccuracy(X,Y):
	predictions = []
	i = 0
	for x in X:
		p = alg.predict([x]).astype(float)
		predictions.append(p[0])


	false_positives = float(sum([1 if p != y and p == 1 else 0 for p,y in zip(predictions,Y)]))
	correct = float(sum([1 if p == y else 0 for p,y in zip(predictions,Y)]))
	accuracy = correct / len (Y)

	print "False Positives: %f" % (float(false_positives)) 
	print "Correct: %f / %f" % (correct, len(Y))
	return accuracy

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
clicks["didBuy"] = clicks["session"].isin(buys["session"])
clicks["didBuy"] = np.where(clicks["didBuy"] == True, 1, 0)

X = clicks.ix[:, 0:4].values
y = clicks.ix[:, 4].values

print "Fitting data..."
alg = linear_model.SGDClassifier(loss="log", penalty="l2")
alg.fit(X,y)

print "Calculating training results..."
print calculateAccuracy(X, y)

print "Loading testing data..."
test_buys = pd.read_csv("data/solution.dat", sep=';', names=buy_headers)
test_clicks = pd.read_csv("data/yoochoose-test.dat", dtype=float, names=click_headers, converters={"category": convertCategory, "time": convertTime})

print "Formatting testing data..."
test_clicks["didBuy"] = test_clicks["session"].isin(test_buys["session"])
test_clicks["didBuy"] = np.where(test_clicks["didBuy"] == True, 1, 0)

t_X = test_clicks.ix[:, 0:4].values
t_y = test_clicks.ix[:, 4].values

print "Calculating testing results..."
print calculateAccuracy(t_X, t_y)

