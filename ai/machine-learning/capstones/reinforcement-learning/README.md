# Machine learning implementation

## Overview

This is an implementation of reinforcement learning using the gymnasium library

## Get started

1. Start venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt
```

2. Start the application
```bash
python3 src/main.py
```

This will start training of the agent and will run five episodes with the agent trained. 

## Notes
- Current implementation has epsilon_policy and q_agent tightly coupled. To improve this in a future consider using interfaces.
