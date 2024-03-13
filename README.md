## SE-PFN Repo

Working version for simple PFN-SE. 
Trained on Synthetic NNEnv prior able to solve simple environments.

Provided with 1000 steps of context on unseen environment, able to clone behaviour of that environment.
Using the original Environment to get initial states, the pfn then can "simulate" the steps taking by an agent.
Using this we can train an RL agent in an online fashion on the PFN using 1000 steps on the real environment.

Examples:

Simple Grid world

And Cartpole