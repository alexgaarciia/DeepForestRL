# DeepForestRL
The goal of this repository is to develop deep reinforcement learning systems that use random forests instead of neural networks. It focuses on building, testing, and optimizing reinforcement learning models with random forests to assess their performance and applicability in various scenarios.

## Files available in the repository
- `main.R`: This R script is the core file in the repository. It includes several key components for the experiment with Reinforcement Learning (RL) and Random Forests in a simulated grid world environment. Here's what each section of the script does:
  1. **Load Necessary Packages**: The script starts by ensuring that the `randomForest` package is installed and loaded, preparing the environment for the Random Forest modeling.
  2. **Define and Initialize the Grid World Setup**: A function (`grid_world`) is defined to create a grid layout, where the agent's goal is to navigate from one point to another. The grid is initialized with dimensions and a specified goal state. In the example described in the file, it is a 5x5 grid, as follows:
  <p align = "center">
     <img src="https://github.com/alexgaarciia/DeepForestRL/blob/main/images/initial_scenario.png" width = 600>
  </p> 

  3. **Generate Simulated Data for State-Action Pairs**: This function (`simulate_data`) generates training data by simulating movement actions within the grid. It logs state-action pairs along with their corresponding rewards, simulating how an agent might explore the environment.
  4. **Train a Random Forest Model**: The script uses the generated data to train a Random Forest model. This model learns to predict the reward outcomes based on different state-action combinations, essentially learning to estimate the value of each action in each state.
  5. **Action Selection Function**: A function (`predict_best_action`) is included to determine the optimal action for any given state based on the predictions from the Random Forest model. This mimics an RL agent deciding which move to make next to maximize its expected reward.
  6. **Test the Model**: Finally, the script tests the trained model's performance in selecting actions from a given state, displaying the recommended action to reach the goal efficiently.
