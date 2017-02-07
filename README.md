# Implementation of A3C Reinforcement Learning Agent in Blocks.

This is an implementation of the A3C-FF and A3C-LSTM agent using the [Blocks library](https://github.com/mila-udem/blocks). A3C is an original work of Google Deepmind you could find in ["Asynchronous Methods for Deep Reinforcement Learning"](http://arxiv.org/pdf/1602.01783v1.pdf)

With this project you can train agents which learn on its own how to play games of the Atari 2600 (and more...), using the Gym environment.


This project was initially intended as a personal project to learn the Blocks library and to code the basic version of an A3C agent at the same time. Keeping this in mind I've tried:
* To reuse as much existent blocks as possible in building the agent.
* When building new blocks could not be avoided, I've tried to use the same coding style of the original library (e.g. the Aysnchronous version of RMSProp).
* To use the same network/parameterization of the A3C agent used in the original paper, although using the gym environment instead of directly interacting with the ALE environment [](https://github.com/mgbellemare/Arcade-Learning-Environment)

Here are some nice gifs of the agent playing Breakout and Phoenix.

![](https://cloud.githubusercontent.com/assets/7602074/22707857/6eea5cc8-ed74-11e6-8ed0-d49bd0fe2177.gif) ![](https://cloud.githubusercontent.com/assets/7602074/22707858/6ef0ff88-ed74-11e6-8f2f-9be931966f71.gif)

## Requirements
* Blocks (http://blocks.readthedocs.io/en/latest/setup.html)
* Theano (http://deeplearning.net/software/theano/install.html)
* Numpy
* Scikit-image: pip install scikit-image
* [gym](https://github.com/openai/gym#installation)
* [gym's atari environment] (https://github.com/openai/gym#atari)

## Usage
In the run_a3c.py file you could find the basic configuration of the A3C agent (gym game, batch_size, image size, etc). To override this configuration use directly the parameters at command line that you can find in a3c_main.py

### Testing

To perform a gym evaluation of a brainless agent just type

```
python run_a3c.py
```
NOTE: check that the log folder or the folder you use to store the results exists before launching the test. You could set a new evaluation folder in the parameter RESULTS_FILE of run_a3c.py.

To perform a gym evaluation of a trained agent just type

```
python run_a3c.py --load_file Breakout-v0_28117541.tar
```

### Training

Just type this command (and be patient). Training a model could perfectly last one day or more... In the paper the agent was trained for 80 million frames in less than one day. 

```
python run_a3c.py --training
```
A checkpoint is saved after several iterations that can be configured at launch time. The checkpoint indicates the number of frames the agent has executed Game-frames.tar (e.g. Breakout-v0_28117541.tar) . 

You could override the game by adding the --game parameter

```
python run_a3c.py --training --game Pooyan-v0
```

NOTE: remember to use the same --game parameter during evaluation


To train an A3C-LSTM agent (not widely tested yet) type:

```
python run_a3c.py --training --a3c_lstm
```

Again, remember to use the same parameters during evaluation

## Results
TODO: Coming soon!


## Other Resources
These links also implement or offer comprehensive explanations of A3C agents. All of them are worthy resources that really deserve taking a look at them.
* [Asyncronous RL in Tensorflow + Keras + OpenAI's Gym](https://github.com/coreylynch/async-rl)
* [A3C implementation at Tensorpack](https://github.com/ppwwyyxx/tensorpack)
* [Async-RL](https://github.com/muupan/async-rl)  . Uses ALE instead of Gym and the original configuration of Deepmind's paper.
* [Kerlym](https://github.com/osh/kerlym)
* Amazing Karpathy's blog entry [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

and many more...

## Additional Notes

* This project implements an Asynchronous version of RMSProp to follow the original paper. Nevertheless Theano does not allow thread-unsafe operations. Hence, each worker captures a lock before updating the network parameters of the shared model, so the final implementation is not completely asynchronous. 

* Gym Atari environment and ALE are a bit different. Gym does not consider life loss as a terminal state whereas the deepmind paper actually does (during training). This will make some differences in the final results.

When building this project I have found particularly important to use Xavier initialization of the network parameters.

Any feedback would be greatly appreaciated if you try this code. And have fun!
