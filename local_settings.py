import os

global base_path
global log_folder
global wandb_project
global openml_path

if "TABPFN_CLUSTER_SETUP" in os.environ:
    if os.environ["TABPFN_CLUSTER_SETUP"] == "FREIBURG":
        print("Using Freiburg Cluster Setup")
        base_path = os.path.join("/work/dlclarge1/hollmann-PFN_Tabular/results_2023")
        log_folder = os.path.join(base_path, "log_test/%j")
        wandb_project = "TabPFN"
        # base_path = os.path.join('/work/dlclarge1/hollmann-PFN_Tabular/results_2023')
        openml_path = os.path.join("/work/dlclarge1/hollmann-PFN_Tabular/results_2023")
    elif os.environ["TABPFN_CLUSTER_SETUP"] == "CHARITE":
        print("Using Charite Cluster Setup")
        base_path = os.path.join("regression")
        log_folder = os.path.join(base_path, "log_test/%j")
        wandb_project = "TabPFN"
        # base_path = os.path.join('/work/dlclarge1/hollmann-PFN_Tabular/results_2023')
        openml_path = os.path.join("/work/dlclarge1/hollmann-PFN_Tabular/results_2023")
    elif os.environ["TABPFN_CLUSTER_SETUP"] == "UNITTEST":
        base_path = "/tmp/prior_fitting_unittests"
        os.makedirs(base_path, exist_ok=True)
        log_folder = os.path.join(base_path, "log_test/%j")
        os.makedirs(log_folder, exist_ok=True)
        wandb_project = "TabPFN"
        openml_path = "/tmp/openml_cache"  # os.path.join("/work/dlclarge1/hollmann-PFN_Tabular/results_2023")
        os.makedirs(openml_path, exist_ok=True)
    else:
        raise ValueError("Unknown Cluster Setup")
    os.makedirs(f"{base_path}/results", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/multiclass", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/regression", exist_ok=True)
    os.makedirs(f"{base_path}/results/tabular/survival", exist_ok=True)
    os.makedirs(f"{base_path}/results/models_diff", exist_ok=True)
else:
    raise ValueError("Unknown Cluster Setup, set TABPFN_CLUSTER_SETUP env variable")
