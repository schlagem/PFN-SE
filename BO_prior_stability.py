import logging
import argparse
import psutil
import pickle
import time

import os

import hpbandster.core.nameserver as hpns
from ConfigSpace import ConfigurationSpace
from ConfigSpace import UniformIntegerHyperparameter
from ConfigSpace import UniformFloatHyperparameter
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace import EqualsCondition
import hpbandster.core.result as hpres

from hpbandster.optimizers import BOHB as BOHB
from bnn_stability_worker import BNN_worker

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(description='BNN debug')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=50)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=50)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=8)
parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str,
                    help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
# parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.') # not needed
parser.add_argument('--shared_directory', type=str,
                    help='A directory that is accessible for all processes, e.g. a NFS share.')



args = parser.parse_args()

# Every process has to lookup the hostname
addrs = psutil.net_if_addrs()
if 'eth0' in addrs.keys():
    print('FOUND eth0 INTERFACE')
    nic = 'eth0'
elif 'eno1' in addrs.keys():
    print('FOUND eno1 INTERFACE')
    nic = 'eno1'
elif 'ib0' in addrs.keys():
    print('FOUND ib0 INTERFACE')
    nic = 'ib0'
else:
    print('FOUND lo INTERFACE')
    nic = 'lo'
#'127.0.0.1'host = hpns.nic_name_to_host(nic)
host = '127.0.0.1'

# python BO_prior_stability.py --worker True --run_id bnn --shared_directory ./
# python BO_prior_stability.py --min_budget 20 --max_budget 100 --n_iterations 100 --n_workers 1 --run_id bnn --shared_directory ./

def get_cs_space():
    cs = ConfigurationSpace()
    cs.add_hyperparameter(UniformIntegerHyperparameter("depth", lower=2, upper=6))
    cs.add_hyperparameter(UniformIntegerHyperparameter("width", lower=4, upper=130))

    cs.add_hyperparameter(UniformFloatHyperparameter("noise", lower=0.0001, upper=3.))
    cs.add_hyperparameter(UniformFloatHyperparameter("init", lower=0.01, upper=10.))

    cs.add_hyperparameter(CategoricalHyperparameter("bias", choices=[True, False]))

    cs.add_hyperparameter(CategoricalHyperparameter("dropout", choices=[True, False]))
    cs.add_hyperparameter(UniformFloatHyperparameter("p_drop", lower=0.1, upper=0.9))
    cs.add_condition(EqualsCondition(cs['p_drop'], cs['dropout'], True))

    cs.add_hyperparameter(CategoricalHyperparameter("res", choices=[True, False]))

    cs.add_hyperparameter(CategoricalHyperparameter("act", choices=["relu", "leaky", "elu", "sin", "identity", "tanh"]))
    return cs


if args.worker:
    # Longer sleep as cluster might need to start nodes
    time.sleep(1)   # short artificial delay to make sure the nameserver is already running
    w = BNN_worker(run_id=args.run_id, host=host)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    print("Worker starting at:", time.strftime("%Y%m%d-%H%M%S"))
    w.run(background=False)
    exit(0)

# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Run an optimizer
# We now have to specify the host, and the nameserver information
result_logger = hpres.json_result_logger(directory=args.shared_directory,
                                         overwrite=True)

bohb = BOHB(configspace=get_cs_space(),
            run_id=args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            min_budget=args.min_budget, max_budget=args.max_budget,
            result_logger=result_logger,
            num_samples=15
            )

print("##################", args.n_iterations)
print("Master starting at:", time.strftime("%Y%m%d-%H%M%S"))
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)


# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()