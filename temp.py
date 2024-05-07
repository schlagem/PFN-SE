#import pygame
#import math
#import numpy as np
import torch.nn as nn
import torch
from owsm_worker import cat_encoder_generator_generator, mlp_encoder_generator_generator
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter
from ConfigSpace import UniformFloatHyperparameter
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import EqualsCondition



def get_cs_space():
    cs = ConfigurationSpace()

    cs.add_hyperparameter(CategoricalHyperparameter("env_name", choices=["NNEnv", "MomentumEnv"]))

    cs.add_hyperparameter(CategoricalHyperparameter("use_bias", choices=[True, False]))

    cs.add_hyperparameter(CategoricalHyperparameter("use_dropout", choices=[True, False]))
    cs.add_hyperparameter(UniformFloatHyperparameter("dropout_p", lower=0.1, upper=0.9))
    cs.add_condition(EqualsCondition(cs['dropout_p'], cs['use_dropout'], True))

    cs.add_hyperparameter(CategoricalHyperparameter("relu", choices=[True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter("sin", choices=[True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter("sigmoid", choices=[True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter("tanh", choices=[True, False]))

    cs.add_hyperparameter(UniformFloatHyperparameter("state_scale", lower=1., upper=20.))
    cs.add_hyperparameter(UniformFloatHyperparameter("state_offset", lower=1., upper=5.))

    cs.add_hyperparameter(CategoricalHyperparameter("use_layer_norm", choices=[True, False]))

    cs.add_hyperparameter(CategoricalHyperparameter("use_res_connection", choices=[True, False]))

    # Encoder Hps
    cs.add_hyperparameter(CategoricalHyperparameter("encoder_res_connection", choices=[True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter("encoder_use_bias", choices=[True, False]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("encoder_depth", lower=1, upper=6))
    cs.add_hyperparameter(CategoricalHyperparameter("encoder_width", choices=[16, 64, 256, 512]))
    cs.add_hyperparameter(CategoricalHyperparameter("encoder_activation", choices=["relu", "sigmoid", "gelu"]))
    cs.add_hyperparameter(CategoricalHyperparameter("encoder_type", choices=["mlp", "cat"]))

    # Decoder Hps
    cs.add_hyperparameter(CategoricalHyperparameter("decoder_res_connection", choices=[True, False]))
    cs.add_hyperparameter(CategoricalHyperparameter("decoder_use_bias", choices=[True, False]))
    cs.add_hyperparameter(UniformIntegerHyperparameter("decoder_depth", lower=1, upper=6))
    cs.add_hyperparameter(CategoricalHyperparameter("decoder_width", choices=[16, 64, 256, 512]))
    cs.add_hyperparameter(CategoricalHyperparameter("decoder_activation", choices=["relu", "sigmoid", "gelu"]))
    cs.add_hyperparameter(CategoricalHyperparameter("decoder_type", choices=["mlp", "cat"]))

    return cs


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
    cs = get_cs_space()
    for i in range(10):
        cfg = cs.sample_configuration()
        fixed_hps = {"num_hidden": 1,
                     "width_hidden": 16,
                     "test": False}

        hps = {**cfg, **fixed_hps}

        if hps["encoder_type"] == "mlp":
            gen = mlp_encoder_generator_generator(hps)
        elif hps["encoder_type"] == "cat":
            gen = cat_encoder_generator_generator(hps, target=False)
        gen = cat_encoder_generator_generator(hps, target=True)
        # print(gen)
        model = gen(14, 512)
        o = model(torch.rand((1001, 4, 14)))
        print(o)
    # print(model)
    # print(model(torch.rand((1001, 4, 14))))

