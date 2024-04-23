from priors.rl_prior import *
import matplotlib.pyplot as plt

env = NNEnvironment()

batch = get_batch(1, 1500, 7, hyperparameters={"test": False})
X, Y = batch.x, batch.y


x = X[:, 0, 0]
y = Y[:, 0, 0]
a = X[:, 0, -1]


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x, a, y, marker="o")
ax.set_xlabel('S Label')
ax.set_ylabel('A Label')
ax.set_zlabel('NS Label')

plt.show()