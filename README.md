# Reinforcement Learning — FrozenLake (Q-Learning + Policy Iteration) & Atari DQN

## Structure
```
RL_Assignment/
├─ README.md
├─ requirements.txt
├─ Part1_FrozenLake/
│  └─ FrozenLake_Qlearning_and_PolicyIteration.ipynb
└─ Part2_AtariDQN/
   └─ Atari_DQN_Breakout.ipynb
```


### FrozenLake
- Implement Q-Learning on `FrozenLake-v1`
- Train & evaluate
- Hyperparameter impact: sweeps of **alpha**, **gamma**, **epsilon** (≥ 3 values each) with discussion prompts
- Implement **Policy Iteration** and compare vs Q-Learning
- Plots for training and evaluation

### Atari DQN
- Adapt Q-Learning to **Deep Q-Learning** on an Atari game (default: Breakout)
- Dueling + Double DQN, replay buffer, target network, preprocessing & framestack
- Train & evaluate

---

## Installation
Use a fresh Python 3.10+ environment.

```bash
pip install -r requirements.txt
```

> **Torch**: Install a version matching your system (CUDA/MPS/CPU) from https://pytorch.org/get-started/locally/
> If `pip install torch` fails, use the official command from the site.

### Atari (Breakout) setup
Gymnasium no longer ships ROMs. You must import them once:

```bash
python - <<'PY'
from ale_py.roms import utils
utils.import_roms('/path/to/ROMs')  # folder containing .bin files such as Breakout.bin
print('ROMs imported into ale-py.')
PY
```

Verify availability in Python:
```python
import gymnasium as gym
print([e.id for e in gym.envs.registry.values() if 'Breakout' in e.id])
# Expect: ['ALE/Breakout-v5', ...]
```

If ALE namespace is missing, check:
```bash
pip install "gymnasium[atari]" --upgrade
```

## Running

### Part 1
Open and run all cells in:
```
Part1_FrozenLake/FrozenLake_Qlearning_and_PolicyIteration.ipynb
```
- Adjust `episodes`/sweeps inside the notebook as desired.

### Part 2
Open and run all cells in:
```
Part2_AtariDQN/Atari_DQN_Breakout.ipynb
```
- Make sure ROMs are imported (see above).
- Start with `total_steps=200_000–300_000` for a demo; longer runs (1M+) improve scores.

## Notes
- The notebooks include explanation cells and result cells. If training is slow, reduce episodes/steps during debugging.
