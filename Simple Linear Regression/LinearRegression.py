
class LinearRegression:
    
    #initialize this object with the training and testing data
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.m = 0
        self.b = 0
    
    #this method will train the data, given a learning rate and number of training iterations
    #this will return the values m and b from the equation y = mx + b
    #with those values, you have your linear regression equation
    #this training algorithm is called gradient descent
    #also prints the cost after every iteration, so you can see if your model is imporving or not
    def train(self, learningRate, iterations):
        m = 0
        b = 0
        for curr in range(iterations):
            
            #cost function
            cost = 0
            for i in range(len(self.X_train)):
                cost += pow(self.Y_train[i] - (m * self.X_train[i] + b), 2)
            cost = cost / len(self.X_train)
            #print out the cost function so you can see what the cost is after every 1000 iterations
            print("Cost after " +  str(curr+1) + " iterations is " + str(cost))
            
            #gradient descent algorithm
            partialM = 0
            partialB = 0
            for i in range(len(self.X_train)):
                partialM += self.X_train[i] * (self.Y_train[i] - (m * self.X_train[i] + b))
                partialB += (self.Y_train[i] - (m * self.X_train[i] + b))
            partialM = (-2 * partialM) / len(self.X_train)
            partialB = (-2 * partialB) / len(self.X_train)
            
            m = m - (learningRate * partialM)
            b = b - (learningRate * partialB)
        
        self.m = m
        self.b = b
        return m, b
    
    #prints the predicted and actual model
    def check(self):
        for i in range(len(self.X_test)):
            y_predicted = (self.m * self.X_test[i])  + self.b
            print("Y Predicted = " + str(y_predicted))
            print("Y Actual    = " + str(self.Y_test[i]))
                