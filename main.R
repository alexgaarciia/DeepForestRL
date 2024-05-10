################################################################################
# This is a simple example in which the agent has to reach point B from point A
# with maximum reward.

# The random forest will be used to predict rewards and assist in action
# selection based on generated data.
################################################################################

# Load necessary packages
install.packages("randomForest")
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
  
  # Loop to simulate state-action pairs and rewards
  for (i in 1:num_samples) {
    # Random initial state
    state <- states[sample(1:nrow(states), 1),]
    
    # Select a random action from the list of available actions
    action <- sample(actions, 1)
    
    # Apply action to calculate next state and reward
    next_state <- state
    
    # Update the row or column of the next state based on the selected action
    if (action == 'up') next_state$row <- max(1, state$row - 1)      # Move up one row
    if (action == 'down') next_state$row <- min(nrow(grid), state$row + 1)  # Move down one row
    if (action == 'left') next_state$col <- max(1, state$col - 1)    # Move left one column
    if (action == 'right') next_state$col <- min(ncol(grid), state$col + 1) # Move right one column
    
    # Determine the reward based on the next state's position in the grid
    # If the agent reaches the goal (last cell), reward = 10; otherwise, reward = -1
    reward <- ifelse(grid[next_state$row, next_state$col] == 1, 10, -1)
    
    # Add the state-action pair and its reward to the data frame
    data <- rbind(data, c(state$row, state$col, action, reward))
  }
  colnames(data) <- c('row', 'col', 'action', 'reward')
  return(data)
}

# STEP 3: Generate training data
training_data <- simulate_data(grid, 500)

# STEP 4: Train a random forest model to predict rewards based on state-action pairs
# STEP 5: Action selection