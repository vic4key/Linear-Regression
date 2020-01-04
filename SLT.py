# -*- coding: utf-8 -*-

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@Author : Vic P.
@Email  : vic4key@gmail.com
@Name   : Manual Linear Regression for Kaggle SLR
@Url    : https://www.kaggle.com/kalcal/simple-linear-regression
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as np
import matplotlib.pyplot as plt
from PyVutils import Others

def Print(A, B, Xs, Ys):

    F = lambda x : B * x  + A # Fx = βx + α

    plt.title(r"Simple Linear Regression (F = $\beta x + \alpha $ = %.3fx + %.3f)" % (A, B))
    plt.axis([min(Xs), max(Xs), min(Ys), max(Ys)])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.plot(Xs, Ys, "o")
    plt.plot(Xs, F(Xs))
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

    return

def LinearRegression(Xs, Ys):

    #  Fx = βx + α

    #  Fx = βx                                      # suppose α = 0 for simplicity

    #  J  = ∑(F𝗑ᵢ - yᵢ)² = ∑((α + β*xᵢ) - yᵢ)²      # the function J called the cost/lost function

    #  J  = ∑(F𝗑ᵢ - yᵢ)² = ∑(βxᵢ - yᵢ)² = ∑(β²xᵢ² - 2βxᵢyᵢ - y²)

    #  J' = ∑(2βxᵢ² - 2xᵢyᵢ) = ∑2βxᵢ² - ∑2xᵢyᵢ      # derivatives J to find the slope

    # ½J' = ∑βxᵢ² - ∑xᵢyᵢ

    # ½J' = 0 <=> ∑βxᵢ² - ∑xᵢyᵢ = 0                 # finding the slope of J

    # β   = ∑xᵢyᵢ ÷ ∑xᵢ²

    # α   = Fxᵢ - βxᵢ (xᵢ & yᵢ here are mean of x & y)

    nPairs = min(np.size(Xs), np.size(Ys)) # pairs of data

    meanX, meanY = np.mean(Xs), np.mean(Ys)

    totalXY = totalXX = 0.

    for i in range(0, nPairs):
        totalXY += (Xs[i] - meanX) * (Ys[i] - meanY)  # ∑xᵢyᵢ
        totalXX += (Xs[i] - meanX) * (Xs[i] - meanX)  # ∑xᵢ²
    pass

    B = totalXY / totalXX   # β = ∑xᵢyᵢ ÷ ∑xᵢ²
    A = meanY - B * meanX   # α = Fxᵢ - βxᵢ (xᵢ & yᵢ here are mean of x & y)

    return (A, B)   # α & β

def main():

    data = np.loadtxt("data/train.csv", delimiter=",", skiprows=1)
    Xs, Ys = list(data.T)
    A, B = LinearRegression(Xs, Ys) # calculate coefficients α & β
    Print(A, B, Xs, Ys)

    data = np.loadtxt("data/sample.csv", delimiter=",", skiprows=1)
    Xs, Ys = list(data.T)
    A, B = LinearRegression(Xs, Ys) # calculate coefficients α & β
    Print(A, B, Xs, Ys)

    return

if __name__ == "__main__":
    try: main()
    except (Exception, KeyboardInterrupt): Others.LogException(sys.exc_info())
    sys.exit()