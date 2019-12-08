import pandas as pd

from visualization import *
from chameleon import *

#making events
start_clock = cuda.Event()
end_clock = cuda.Event()

if __name__ == "__main__":
    
    start_clock.record()

    # get a set of data points
    df = pd.read_csv('./datasets/two_squares.csv', sep=' ',
                     header=None)

    # returns a pandas.dataframe of cluster
    res = cluster(df, 7, knn=6, m=40, alpha=2.0, plot=False)

    # draw a 2-D scatter plot with cluster
    plot2d_data(res)
    
    end_clock.record()
    end_clock.synchronize()
    time = start.time_till(end)*1e-3
    print "%fs" % (secs)
