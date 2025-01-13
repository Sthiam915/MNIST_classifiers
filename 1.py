import statistics as s

x = [1.023e13, 8.218e11, 6.429e11, 5.276e11]


print("mean: {:.2e} stdev: {:.2e}".format(s.mean(x), s.stdev(x)))