# Lunar Landing Project

Made for TCSS 435 by:

- [Charankamal Brar](https://github.com/csbrar25)
- [Johan Hernandez](https://github.com/johan253)
- [Brittney Jones](https://github.com/jonesb7)
- [Lucas Perry](https://github.com/lperry2)
- [Corey Young](https://github.com/cyoung5233)

## Setup

In order to run this project, you need to have python `^3.11` installed. The `pip` requirements can be found in `requirements.txt` and can be installed by running:

```bash
pip install -r requirements.txt
```

After the dependencies are installed, you can run the project by running:

```bash
python main.py
```

The results of the (each) DQN agent will be:

1. Displayed on the final episode of training as a popup window
2. Exported as graphs and tables in the `results` folder

## DQN Extension

For the DQN extension our group went with the **Double DQN**. When running the program, it will default to running it twice: Once for the single vanilla DQN, and another time for the double DQN version. In order to only run the vanilla DQN, you must run the program with the `--run_one` flag.

```bash
python main.py --run_one
```

In order to run the program with only the _double_ DQN, you must pass the `--double` flag along with the `--run_one` flag.

```bash
python main.py --run_one --double
```

The results of any running the program will be saved into the `results` folder, which will by default generate a graph for comparing the vanilla vs the double DQN agents in one graph.
