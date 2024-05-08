#import pygame
#import math
#import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from priors.rl_prior import generate_bnn
from priors.rl_prior import get_bnn_train_batch
from priors.rl_prior import get_bnn_batch
from priors.rl_prior import AdditiveNoiseLayer
import time
import tqdm
"""
pygame.init()

screen = pygame.display.set_mode((500, 500))

clock = pygame.time.Clock()

rad = 4.7
rad = math.atan2(math.sin(rad), math.cos(rad))
sin = math.sin(rad)
cos = math.cos(rad)
vel = 0.
g = 50.

x = np.random.rand()

while True:
    # Process player inputs.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    # Do logical updates here.
    # ...

    screen.fill("purple")  # Fill the display with a solid color

    # Render the graphics here.
    # ...
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        vel = vel + .5 * 0.1
    elif keys[pygame.K_RIGHT]:
        vel = vel - (.5 * 0.1)

    # vel -= g * math.cos(rad) * 0.01
    vel *= 0.99
    max_vel = 7
    if vel > max_vel:
        vel = max_vel

    if vel < -max_vel:
        vel = -max_vel

    rad += vel * 0.1
    print("----")
    print(math.cos(math.atan2(math.sin(rad), math.cos(rad))))
    print(math.sin(math.atan2(math.sin(rad), math.cos(rad))))
    print(math.atan2(math.sin(rad), math.cos(rad)))

    x = math.cos(rad)_
    y = math.sin(rad)
    print("x:", x)
    print("y:", y)
    pygame.draw.line(screen, "green", (250, 250), (250 + 100 * x, 250 - 100 * y), 5)
    max_vel = 7.
    if vel > max_vel:
        vel = max_vel

    if vel < -max_vel:
        vel = -max_vel


    x += vel * 0.1

    max_pos = 1.
    if x > max_pos:
        x = max_pos
        vel = -.7 * vel

    min_pos = 0.
    if x < min_pos:
        x = min_pos
        vel = -.7 * vel

    print(vel)

    pygame.draw.rect(screen, "green", pygame.Rect(225, 450 - x * 450, 50, 50))
    pygame.display.flip()  # Refresh on-screen display
    clock.tick(30)         # wait until next frame (at 60 FPS)
"""
if __name__ == '__main__':
    t = []
    for i in range(100):
        s = time.time()
        b = get_bnn_batch(4, 1001, 14)
        t.append(time.time()-s)
    print(max(t))
    print(min(t))
    print(sum(t)/len(t))

