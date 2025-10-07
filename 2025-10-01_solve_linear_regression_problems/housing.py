import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_w1(x, r):
    x_bar = np.mean(x)
    r_bar = np.mean(r)
    return (np.mean(x * r) - x_bar * r_bar) / (np.mean(x ** 2) - x_bar ** 2)

def get_w0(x, r):
    return np.mean(r) - get_w1(x, r) * np.mean(x)

def f_of_x(x, w1, w0):
    return w1 * x + w0

if __name__ == "__main__":
    features = [
        "per capita crime rate by town(CRIM)",
        "proportion of residential land zoned for lots over 25,000 sq.ft(ZN)",
        "proportion of non-retail business acres per town(INDUS)",
        "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)[CHAS]",
        "nitric oxides concentration (parts per 10 million)[NOX]",
        "average number of rooms per dwelling(RM)",
        "proportion of owner-occupied units built prior to 1940(AGE)",
        "weighted distances to five Boston employment centres(DIS)",
        "index of accessibility to radial highways(RAD)",
        "full-value property-tax rate per $10,000(TAX)",
        "pupil-teacher ratio by town(PTRATIO)",
        "1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town[B]",
        "% lower status of the population(LSTAT)",
        "Median value of owner-occupied homes in $1000's(MEDV)"
    ]
    df = pd.read_csv("housing.data", sep='\s+', names=features)

    s = int(len(df) * 0.8)

    for feature in features[:-1]:
        print(f"Features: {feature}")
        x = df[feature].values
        r = df["Median value of owner-occupied homes in $1000's(MEDV)"].values
        w1 = get_w1(x, r)
        w0 = get_w0(x, r)
        print(f"w1: {w1:.2f}, w0: {w0:.2f}")
        print(f"f(x)={w1:.2f}x + {w0:.2f}")
        fig = plt.figure()
        plt.scatter(x, r, color='lightpink', edgecolors='darkolivegreen')
        plt.plot(x, f_of_x(x, w1, w0), 'r-')
        plt.xlabel(feature)
        plt.ylabel("Median value of owner-occupied homes in $1000's(MEDV)")
        plt.savefig(f"model/MEDV-{feature}.png")
        print("---")


