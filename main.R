################################################################################
# We will consider a basic reinforcement learning task where the agent must
# choose between two actions at each step, with each action having a different
# probability of success.

# The random forest will be used to predict the expected reward from each action
# based on past data.
################################################################################

# Install necessary packages
install.packages("randomForest") # Implements Breiman's random forest algorithm for classification and regression
install.packages("dplyr") # Package for making data manipulation easier 
library(randomForest)
library(dplyr)

# Simulate some data (each row represents a state-action pair and its outcome)
set.seed(123)
data <- data.frame(
  state = rep(1:10, each=20), # 10 states
  action = rep(1:2, 100), # 2 possible actions
  reward = rnorm(200, mean=5, sd=2) + rep(c(0,1), each=100) # Random rewards, action 2 slightly better
)

# Split the data (divide the data intro training and testing tests)
train_idx <- sample(1:nrow(data), 0.8*nrow(data))
training <- data[train_idx, ]
testing <- data[-train_idx, ]
