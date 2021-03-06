import bsuite
from bsuite import sweep
from bsuite.baselines import experiment
from bsuite.baselines.utils import pool

from agent import Agent
import torch

save_path = "./runs/run032/"  # Path were the results are saved.


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
    qnet_settings = {'layers_sizes': [50], 'batch_size': 64, 'noisy_nets': True, 'distributional': True, 'vmin': 0,
                     'vmax': 1000, 'number_atoms': 51}

    # Settings for the specific agent
    settings = {'batch_size': qnet_settings["batch_size"], 'epsilon_start': 0.0, 'epsilon_decay': 0.00,
                'epsilon_min': 0.00, 'gamma': 0.99, 'buffer_size': 2 ** 16, 'lr': 1e-3, 'qnet_settings': qnet_settings,
                'start_optimization': 64, 'update_qnet_every': 2, 'update_target_every': 50, 'ddqn': True, 'n_steps': 4,
                'duelling_dqn': True, 'prioritized_buffer': True, 'alpha': 0.6, 'beta0': 0.4, 'beta_increment': 1e-6}

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


bsuite_sweep = getattr(sweep, 'CARTPOLE_SCALE')
pool.map_mpi(run, bsuite_sweep, 6)


#bsuite_sweep = getattr(sweep, 'SWEEP')
#pool.map_mpi(run, bsuite_sweep, 6)
