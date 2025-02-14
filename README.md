A Tetris-playing reinforcement learning agent that utilizes a neural network trained via Double Deep Q-Learning with Prioritized Experience Replay, implemented in PyTorch. For more details, see:

[Deep Q-Learning](https://arxiv.org/abs/1312.5602)

[Double Deep Q-Learning](https://arxiv.org/abs/1509.06461)

[Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)

A simple Tetris clone was also created for the agent to actually play in; a demo of the agent and the environment can be found [here](https://youtu.be/qtfc6Nf5_zc)

## Implementation

Unlike a usual Q-Network, the network here only has one output. Decisions on placements are only made right after the placement of the previous tetromino; to determine a placement, the program finds all of the legal placements for the current tetromino and what the board would look like in each case. Features are computed for each possible board and the network uses them to estimate a Q-value for each outcome; so, for each tetromino placement, the network is computed multiple times. Then, the program performs the moves that result in the board with the greatest estimated Q-value. To use a 1-piece lookahead, the program finds all of the possible placements of the current and next tetrominoes, the Q-values are estimated, and the moves are performed to place the current tetromino in the desired location; the process is repeated for the next tetromino. If one thinks of each possible future board after placing one tetromino as possible "actions", then the network is trained in the usual TD-error manner of a Double Q-Network, aside from the special sampling scheme.

The network takes 12 features as inputs: the heights of each of the 10 columns, the number of holes in the board (where a "hole" is defined as an open tile in the board with a filled tile anywhere above it), and the number of line clears gained in the transition between boards. This is followed by two dense layers of 400 and 300 nodes each with rectifier activations, before the 1-node output layer.

## Results

As far as I can tell, using lookahead, the agent can play forever (specifically, the 2 million-steps pretrained network; I have not extensively tested the others). It was able to play up to around 200,000 lines cleared before I had to stop it due to time constraints. The pretrained networks with more steps get better "scores", as in, they get more triples and tetrises and fewer singles and doubles, though this also means they play more recklessly, so they likely cannot consistently play as long as the network with 2 million steps.

## Personal Use

There are three folders in the "saves" folder that characterize three networks acquired during various stages of training. The folders contain pickled files with the target and online network weights, the RAdamW optimizer weights, the agent parameters, and the experience replay cache (though, the cache has been emptied in these uploaded folders due to size considerations). The networks can be used by downloading the saves folder along with the notebook file, putting the two in the same folder, and running the cells; see the last cell for further directions on using the networks or training your own. The weights can be used directly as state dictionaries in PyTorch, with considerations to network dimensions.
