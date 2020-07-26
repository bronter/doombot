# doombot
Bot that plays DOOM levels

Currently a work-in-progress, just ported from the OpenAI Gym: https://github.com/ppaquette/gym-doom (no longer maintained)

This is a neural-network based agent that learns to play doom, using something similar to an actor-critic setup where the screen content first is fed through a series of 2d convolution and max pooling layers, then gets fed to two entities:
* The Agent - This is a fully-connected LSTM connected to the convolutional layers output, it puts out an array of actions which we can read from to take an action in the game, and they're also fed to the second part of the network, the judge.
* The Judge - Another fully-connected LSTM, this one takes both the actions from the agent and the output of the convolutional layers, and tries to predict what the score (well, really the reward) will be once the action has been taken.

I train the network by running two different optimization passes which sort of battle each other - The first one tries to minimize the difference between the predicted reward and the actual reward from the game, giving the judge an idea of which actions have which effect on the world, and the second pass tries to maximize the reward, so the agent takes better actions for getting larger rewards.  I've been playing with having the only updating the weights and biases for the agent and convolutional layers on the second pass, but it seems to make little difference, so I update all the weights and biases for both the agent and judge on each pass.

Requirements:
-------------
Tensorflow 1.14

ViZDoom - Helps to also clone their repository if you want to use their pre-made configs and WADs: https://github.com/mwydmuch/ViZDoom/tree/master/scenarios

Numpy
