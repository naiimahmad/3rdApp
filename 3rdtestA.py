# Perform the prediction by using the trained model
step_size = 0.01

# Plot the Decision Boundaries
x_min, x_max = min(X.iloc[:,0]) - 1, max(X.iloc[:,0]) + 1
y_min, y_max = min(X.iloc[:,1]) - 1, max(X.iloc[:,1]) + 1
x_values, y_values = np.meshgrid(np.arange(x_min,x_max,step_size), np.arange(y_min,y_max,step_size))

# Predict labels for all points in the mesh
predictions = kmeans.predict(np.c_[x_values.ravel(), y_values.ravel()]) 

# Plot the results
predictions = predictions.reshape(x_values.s
some code from internet with modification
