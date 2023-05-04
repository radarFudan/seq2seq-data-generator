This dataset is generated by direct numerical simulaition results of a flow fast cylinder case. We simulated flow at Reynolds number = 40,60,100 and 250. Folder with suffix '_pod' contains data reconstructed using 2 dominant POD modes, accounting for >99% energy of the original data.

p1-p6 stand for 6 spatial points. For each point, the data shape is (20001,4), where 20001 means time steps with dt = 0.005s, 4 means 4 variables (t, u, v, p) respectively. The generator is not includes because the training and testing data format can be flexible, e.g, temporal prediction of single variable, or use one variable to predict another.