# -*- coding: utf-8 -*-

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
@Author : Vic P.
@Email  : vic4key@gmail.com
@Name   : Manual LR model for Kaggle SLR
@Url    : https://www.kaggle.com/kalcal/simple-linear-regression
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import sys
import numpy as NP
import matplotlib.pyplot as PLT
import Vutil as vu

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

    nPairs = min(NP.size(Xs), NP.size(Ys)) # pairs of data

    meanX, meanY = NP.mean(Xs), NP.mean(Ys)

    totalXY = totalXX = 0.

    for i in xrange(0, nPairs):
        totalXY += (Xs[i] - meanX) * (Ys[i] - meanY)  # ∑xᵢyᵢ
        totalXX += (Xs[i] - meanX) * (Xs[i] - meanX)  # ∑xᵢ²
    pass

    B = totalXY / totalXX   # β = ∑xᵢyᵢ ÷ ∑xᵢ²
    A = meanY - B * meanX   # α = Fxᵢ - βxᵢ (xᵢ & yᵢ here are mean of x & y)

    return (A, B)   # α & β

def main():

    l = vu.ReadCVS("data/train.csv", start=1, nrows=100)

    pairs = NP.array(l).astype(NP.float)  # NP.random.uniform(low=5, high=10, size=(10, 2)) # NP.random.randn(50, 2)

    Xs, Ys = list(pairs.transpose())      # split to Xs & Ys

    A, B = LinearRegression(Xs, Ys)       # calculate coefficients α & β

    F = lambda x : B * x  + A             # Fx = βx + α

    PLT.title(r"Simple Linear Regression (F = $\beta x + \alpha $ = %.3fx + %.3f)" % (A, B))
    PLT.axis([min(Xs), max(Xs), min(Ys), max(Ys)])
    PLT.xlabel("X")
    PLT.ylabel("Y")
    PLT.grid(True)
    PLT.plot(Xs, Ys, "o")
    PLT.plot(Xs, F(Xs))
    PLT.show()

    return

if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt): vu.LogException(sys.exc_info())
    sys.exit()