import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


class OnlineRandomForest:
    def __init__(self, n_trees=10, min_samples_split=10, min_gain=0.1):
        """
        Initialize the Online Random Forest.

        Parameters:
        - n_trees: Number of trees in the forest.
        - min_samples_split: Minimum number of samples required to split a node.
        - min_gain: Minimum information gain required to split a node.
        """
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.trees = [DecisionTreeClassifier(max_depth=None) for _ in range(n_trees)]
        self.data = [[] for _ in range(n_trees)]  # List to store data for each tree
        self.labels = [[] for _ in range(n_trees)]  # List to store labels for each tree

    def poisson(self, lam=1):
        """
        Generate a Poisson-distributed random number.

        Parameters:
        - lam: Lambda value for the Poisson distribution (default is 1).

        Returns:
        A Poisson-distributed random number.
        """
        return np.random.poisson(lam)

    def update_tree(self, tree_index, X, y):
        """
        Update a specific tree with a new data point.

        Parameters:
        - tree_index: Index of the tree to update.
        - X: Feature vector of the new data point.
        - y: Label of the new data point.
        """
        k = self.poisson()  # Determine how many times to update the tree
        if k > 0:
            for _ in range(k):
                self.data[tree_index].append(X)
                self.labels[tree_index].append(y)

                # If there are enough samples, update the tree
                if len(self.data[tree_index]) > self.min_samples_split:
                    tree = self.trees[tree_index]
                    tree.fit(self.data[tree_index], self.labels[tree_index])

    def update_forest(self, X, y):
        """
        Update all trees in the forest with a new data point.

        Parameters:
        - X: Feature vector of the new data point.
        - y: Label of the new data point.
        """
        for i in range(self.n_trees):
            self.update_tree(i, X, y)

    def predict(self, X):
        """
        Predict the class labels for a set of samples.

        Parameters:
        - X: Feature matrix of the samples to predict.

        Returns:
        Predicted class labels.
        """
        predictions = np.zeros((self.n_trees, len(X)))
        for i, tree in enumerate(self.trees):
            if self.data[i]:  # Only predict if the tree has been trained with data
                predictions[i] = tree.predict(X)
        # Aggregate the predictions from all trees
        return np.round(np.mean(predictions, axis=0))


# Generate synthetic data for demonstration
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2)

# Create an instance of OnlineRandomForest
orf = OnlineRandomForest(n_trees=5)

# Simulate the arrival of data in real-time
for i in range(len(X)):
    orf.update_forest(X[i], y[i])  # Update the forest with the new data point
    if i % 100 == 0:
        print(f"Processed {i + 1} samples")

# Generate synthetic test data
X_test, y_test = make_classification(n_samples=200, n_features=20, n_informative=15, n_classes=2)

# Make predictions on the test data
predictions = orf.predict(X_test)
accuracy = np.mean(predictions == y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")
