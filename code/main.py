import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.utils import pool

from agent import Agent
import torch

save_path = "./runs/run005/"  # Path were the results are saved.


def run(bsuite_id: str) -> str:
    """
    Runs a bsuite experiment and saves the results as csv files

    Args:
        bsuite_id: string, the id of the bsuite experiment to run

    Returns: none

    """
    env = bsuite.load_and_record(
        bsuite_id=bsuite_id,
        save_path=save_path,
        logging_mode='csv',
        overwrite=True,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Settings for the neural network
    qnet_settings = {"layers_sizes": [50], "batch_size": 128}
    # Settings for the specific agent
    settings = {"batch_size": qnet_settings["batch_size"], "epsilon_start": 1.0, "epsilon_decay": 0.999,
                "epsilon_min": 0.01, "gamma": 0.99, "buffer_size": 300000, "lr": 2.5e-4, "qnet_settings": qnet_settings,
                "start_optimization": 128, "update_qnet_every": 2, "update_target_every": 100}

    agent = Agent(action_spec=env.action_spec(),
                  observation_spec=env.observation_spec(),
                  device=device,
                  settings=settings
                  )

    experiment.run(
        agent=agent,
        environment=env,
        num_episodes=env.bsuite_num_episodes,
        verbose=False)
    return bsuite_id


bsuite_sweep = getattr(sweep, 'SWEEP')
pool.map_mpi(run, bsuite_sweep, 6)
