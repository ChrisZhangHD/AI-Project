import numpy as np
import math


class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k
        self.trainX = []
        self.trainY = []

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        # training logic here
        # input is an array of features and labels
        self.trainX = X
        self.trainY = y

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        predictResult = []
        for x1 in X:
            count = 0
            distances = []
            for x2 in self.trainX:
                distances.append(self.distance(x1, x2))
            neighbors = np.argsort(distances)[: self.k]

            for neighbor in neighbors:
                if self.trainY[neighbor] == 1:
                    count += 1
            if count > self.k - count:
                predictResult.append(1)
            elif count == self.k - count:
                predictResult.append(np.random.randint(0, 1))
            else:
                predictResult.append(0)
        return np.asarray(predictResult)


class ID3:
    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range
        self.tree = {}
        self.trainY = []

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def getSubsetRowsByAttributeClass(self, dataset, rows, attribute, classValue):
        subRows = []
        for row in rows:
            if dataset[row][attribute] == classValue:
                subRows.append(row)
        return subRows

    def entropy(self, dataset, rows):
        totalLabel = len(rows)
        count_YES = 0
        count_NO = 0

        for row in rows:
            if dataset[row][-1] == 1:
                count_YES += 1
            else:
                count_NO += 1

        if count_YES == 0:
            return 0
        elif count_NO == 0:
            return 0

        else:
            return - (count_YES / totalLabel) * math.log2(count_YES / totalLabel) \
                    - (count_NO / totalLabel) * math.log2(count_NO / totalLabel)

    def selectBestAttribute(self, dataset, baseEntropy, rows, attributes):
        maxGain = 0
        bestAttribute = -1
        for attribute in attributes:
            attributeBaseEntropy = baseEntropy
            attributeClassDict = {}
            for row in rows:
                classKey = dataset[row][attribute]
                if classKey not in attributeClassDict:
                    attributeClassDict[classKey] = []
                attributeClassDict[classKey].append(row)
            for key in attributeClassDict:
                classEntropy = self.entropy(dataset, attributeClassDict[key])
                attributeBaseEntropy -= (len(attributeClassDict[key]) / len(rows)) * classEntropy
            if attributeBaseEntropy >= maxGain:
                maxGain = attributeBaseEntropy
                bestAttribute = attribute
        return bestAttribute

    def buildTree(self, dataset, rows, attributes):
        decisionLabelList = []
        for row in rows:
            decisionLabelList.append(dataset[row][-1])

        count_YES = np.count_nonzero(decisionLabelList)
        if len(attributes) == 0:
            if count_YES * 2 > len(rows):
                return 1

            else:
                return 0

        if count_YES == 0:
            return 0
        elif count_YES == len(decisionLabelList):
            return 1

        baseEntropy = self.entropy(dataset, rows)
        bestAttribute = self.selectBestAttribute(dataset, baseEntropy, rows, attributes)
        attributes.remove(bestAttribute)
        attributeClasses = set(dataset[row][bestAttribute] for row in rows)
        tree = [bestAttribute, {}]
        for attributeClass in attributeClasses:
            subTree = self.buildTree(dataset, self.getSubsetRowsByAttributeClass(dataset, rows, bestAttribute, attributeClass), attributes)
            tree[1][attributeClass] = subTree
        return tree

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        self.trainY = y
        categorical_data = self.preprocess(X)
        dataset = np.concatenate((categorical_data, np.reshape(y, (-1, 1))), axis=1)
        # entropyDecision = self.entropy(dataset)
        attributesCount = len(dataset[0]) - 1
        attributes = set()
        for i in range(attributesCount):
            attributes.add(i)
        subsetRows = []
        for r in range(len(dataset)):
            subsetRows.append(r)
        self.tree = self.buildTree(dataset, subsetRows, attributes)


    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        res = []
        count = np.count_nonzero(self.trainY)
        if count * 2 >= len(self.trainY):
            defaultLabel = 1
        else:
            defaultLabel = 0
        for i in range(len(categorical_data)):
            curTree = self.tree
            while curTree is not None:
                if not isinstance(curTree, list):
                    res.append(curTree)
                    break
                attributeClass = categorical_data[i][curTree[0]]
                if attributeClass in curTree[1]:
                    curTree = curTree[1][attributeClass]
                else:
                    res.append(defaultLabel)
                    break
        return np.asarray(res)


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b
        self.d = [1, -1]

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        for step in range(steps):
            i = step % len(y)
            predictLabel = np.dot(X[i], self.w) + self.b
            if predictLabel > 0:
                predictLabel = 1
            else:
                predictLabel = 0
            if predictLabel != y[i]:
                if y[i] == 1:
                    self.w = self.w + self.lr * self.d[0] * X[i]
                else:
                    self.w = self.w + self.lr * self.d[1] * X[i]

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        result = []
        for x in X:
            res = np.dot(x, self.w) + self.b
            if res > 0:
                res = 1
            else:
                res = 0
            result.append(res)
        result = np.asarray(result)
        return result


class MLP:
    def __init__(self, w1, b1, w2, b2, lr):
        self.l1 = FCLayer(w1, b1, lr)
        self.a1 = Sigmoid()
        self.l2 = FCLayer(w2, b2, lr)
        self.a2 = Sigmoid()

    def MSE(self, prediction, target):
        return np.square(target - prediction).sum()

    def MSEGrad(self, prediction, target):
        return - 2.0 * (target - prediction)

    def shuffle(self, X, y):
        idxs = np.arange(y.size)
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def train(self, X, y, steps):
        for s in range(steps):
            i = s % y.size
            if (i == 0):
                X, y = self.shuffle(X, y)
            xi = np.expand_dims(X[i], axis=0)
            yi = np.expand_dims(y[i], axis=0)

            pred = self.l1.forward(xi)
            pred = self.a1.forward(pred)
            pred = self.l2.forward(pred)
            pred = self.a2.forward(pred)
            loss = self.MSE(pred, yi)
            # print(loss)

            grad = self.MSEGrad(pred, yi)
            grad = self.a2.backward(grad)
            grad = self.l2.backward(grad)
            grad = self.a1.backward(grad)
            grad = self.l1.backward(grad)

    def predict(self, X):
        pred = self.l1.forward(X)
        pred = self.a1.forward(pred)
        pred = self.l2.forward(pred)
        pred = self.a2.forward(pred)
        pred = np.round(pred)
        return np.ravel(pred)


class FCLayer:

    def __init__(self, w, b, lr):
        self.lr = lr
        self.w = w  # Each column represents all the weights going into an output node
        self.b = b
        self.x = np.zeros((1, 30))

    def forward(self, input):
        # Write forward pass here
        self.x = input
        return np.dot(input, self.w) + self.b

    def backward(self, gradients):
        # Write backward pass here
        wDiff = np.dot(self.x.T, gradients)
        xDiff = np.dot(gradients, self.w.T)
        self.w = self.w - self.lr * wDiff
        self.b = self.b - self.lr * gradients
        return xDiff


class Sigmoid:

    def __init__(self):
        self.y = 0

    def forward(self, input):
        # Write forward pass here
        [rows, cols] = input.shape
        self.y = np.zeros_like(input)
        for i in range(rows):
            for j in range(cols):
                if input[i, j] < 0:
                    self.y[i, j] = np.exp(input[i, j]) / (1 + np.exp(input[i, j]))
                else:
                    self.y[i, j] = 1 / (1 + np.exp(-input[i, j]))
        return self.y

    def backward(self, gradients):
        # Write backward pass here
        sigmoidGradients = gradients * (1-self.y) * self.y
        return sigmoidGradients
