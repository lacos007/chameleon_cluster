import pandas as pd
import pycuda.driver as cuda

from visualization import *
from chameleon import *
import pycuda.driver as cuda


if __name__ == "__main__":

    # get a set of data points
    df = pd.read_csv('./datasets/two_squares.csv', sep=' ',
                     header=None)

    # returns a pandas.dataframe of cluster
    res = cluster(df, 7, knn=6, m=40, alpha=2.0, plot=False)

    # draw a 2-D scatter plot with cluster
    plot2d_data(res)

