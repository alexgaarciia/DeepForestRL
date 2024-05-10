################################################################################
# This is a simple example in which the agent has to reach point B from point A
# with maximum reward.

# The random forest will be used to predict rewards and assist in action
# selection based on generated data.
################################################################################

# Load necessary packages
if (!require('randomForest')) install.packages('randomForest')
library(randomForest)

# STEP 1: Define and initialize the grid world setup
grid_world <- function(size) {
  grid <- matrix(rep(0, size * size), nrow = size)
  grid[size, size] <- 1  # Goal state
  return(grid)
}

grid <- grid_world(5)
grid


# STEP 2: Generate simulated data for state-action pairs
simulate_data <- function(grid, num_samples) {
  # Available actions that the agent can take
  actions <- c('up', 'down', 'left', 'right')
  
  # Create all possible states by combining each row and column in the grid
  states <- expand.grid(row = 1:nrow(grid), col = 1:ncol(grid))
  
  # Initialize an empty data frame to store state-action pairs and their rewards
  data <- data.frame()
  
  # Outer loop: Simulate data for a given number of random starting states
  for (i in 1:num_samples) {
    # Select a random state (row and column index) from the list of possible states
    state <- states[sample(1:nrow(states), 1),]
    
    # Inner loop: For each starting state, explore all actions
    for (action in actions) {
      # Initialize the next state to the current state before applying any action
      next_state <- state
      
      # Update the next state according to the chosen action while ensuring grid boundaries
      if (action == 'up') {
        next_state$row <- max(1, state$row - 1)  # Move up (reduce row index but not below 1)
      } else if (action == 'down') {
        next_state$row <- min(nrow(grid), state$row + 1)  # Move down (increase row index but not above grid limit)
      } else if (action == 'left') {
        next_state$col <- max(1, state$col - 1)  # Move left (reduce column index but not below 1)
      } else if (action == 'right') {
        next_state$col <- min(ncol(grid), state$col + 1)  # Move right (increase column index but not above grid limit)
      }
      
      # Determine the reward based on whether the agent reaches the goal
      # +10 if the agent reaches the goal position; otherwise, -1 (penalty)
      reward <- ifelse(grid[next_state$row, next_state$col] == 1, 10, -1)
      
      # Append the state-action-reward pair to the data frame
      data <- rbind(data, c(state$row, state$col, action, reward))
    }
  }
  colnames(data) <- c('row', 'col', 'action', 'reward')
  return(data)
}


# STEP 3: Generate training data
training_data <- simulate_data(grid, 200)


# STEP 4: Train a random forest model to predict rewards based on state-action pairs
# Convert action to a factor (categorical) type for classification
training_data$action <- as.factor(training_data$action)
training_data$reward <- as.factor(ifelse(training_data$reward > 0, "Positive", "Negative"))

# Train a random forest model using classification
rf_model <- randomForest(reward ~ row + col + action, data = training_data, ntree = 100, importance = TRUE)


# STEP 5: Action selection
# Function to predict the best action for a given state
predict_best_action <- function(model, state) {
  actions <- c('up', 'down', 'left', 'right')
  possible_actions <- data.frame(row = rep(state$row, 4),
                                 col = rep(state$col, 4),
                                 action = factor(actions, levels = actions))
  predictions <- predict(model, possible_actions, type = "prob")[, "Positive"]
  best_action <- actions[which.max(predictions)]
  return(best_action)
}

# Test the model in a random starting state
start_state <- list(row = 5, col = 4)
best_action <- predict_best_action(rf_model, start_state)
cat("Best action from state (", start_state$row, ",", start_state$col, ") is: ", best_action, "\n")
