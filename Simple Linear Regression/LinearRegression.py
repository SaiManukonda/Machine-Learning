
class LinearRegression:
    
    #initialize this object with the training and testing data
    def __init__(self, X_train, Y_train, X_test, Y_test):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    
    #this method will train the data, given a learning rate and number of training iterations
    #this will return the values m and b from the equation y = mx + b
    #with those values, you have your linear regression equation
    #this training algorithm is called gradient descent
    def train(self, learningRate, iterations):
        m = 0
        b = 0
        for i in range(iterations):
            
            #cost function
            cost = 0
            for i in range(len(self.X_train)):
                cost += pow(self.Y_train[i] - (m * self.X_train[i] + b), 2)
            cost = cost / len(self.X_train)
            #print out the cost function so you can see what the cost is after every 1000 iterations
            if iterations % 1000 == 0:
                print("Cost after " +  str(iterations) + " iterations is " + str(cost))
            
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
        return m, b
                