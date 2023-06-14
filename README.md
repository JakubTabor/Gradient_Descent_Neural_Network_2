# Gradient_Descent_Neural_Network_2
I gonna build neural network from scratch also with gradient descent inside 
# I gonna use simple dataset with only three columns to make things simple
# I have my data ready, so I import "train_test_split" and put age and affordibility as X variable and bought_insurance as y variable
# Then for better performance I scaled my "age column" in X_train and X_test but before this i make copy on it 
# So the preprocessing phase was short, now we can build simple network with only one layer
# And then I gonna build my own neural network but from scratch, to show the simplicity of "keras.Sequential" structure
# I my keras model I put last Dense layer with one neuron, two variables (age and affordibility) it is shape, then activation as sigmoid
# I initialize kernel with ones and bias with zeros, I keep the structure simple 
# Then i compile my model with same parameters as always, so optimizer as adam, I have binary output, so loss will be binary_crossentropy
# And I train my model with X_train_scaled and y_train, I set number of epochs at 1000, after evaluating my model I get good accuracy
