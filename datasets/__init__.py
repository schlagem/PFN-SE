import copy

import pandas as pd
import torch
import numpy as np
import openml
import random
from typing import Optional


class DatasetModifications:
    def __init__(self, classes_capped: bool, feats_capped: bool, samples_capped: bool):
        """
        :param classes_capped: Whether the number of classes was capped
        :param feats_capped: Whether the number of features was capped
        :param samples_capped: Whether the number of samples was capped
        """
        self.classes_capped = classes_capped
        self.feats_capped = feats_capped
        self.samples_capped = samples_capped


class TabularDataset:
    def __init__(
        self,
        name: str,
        x: torch.tensor,
        y: torch.tensor,
        task_type: str,
        attribute_names: list[str],
        categorical_feats: Optional[list[int]] = [],
        modifications: Optional[DatasetModifications] = None,
        splits: Optional[list[tuple[torch.tensor, torch.tensor]]] = None,
    ):
        """
        :param name: Name of the dataset
        :param x: The data matrix
        :param y: The labels
        :param categorical_feats: A list of indices of categorical features
        :param attribute_names: A list of attribute names
        :param modifications: A DatasetModifications object
        :param splits: A list of splits, each split is a tuple of (train_indices, test_indices)
        """
        self.name = name
        self.x = x
        self.y = y
        self.categorical_feats = categorical_feats
        self.attribute_names = attribute_names
        self.modifications = (
            modifications
            if modifications is not None
            else DatasetModifications(
                classes_capped=False, feats_capped=False, samples_capped=False
            )
        )
        self.splits = splits
        self.task_type = task_type

    def __getitem__(self, indices):
        # convert a simple index x[y] to a tuple for consistency
        # if not isinstance(indices, tuple):
        #    indices = tuple(indices)
        ds = copy.deepcopy(self)
        ds.x = ds.x[indices]
        ds.y = ds.y[indices]

        return ds

    def generate_valid_split(self, bptt, eval_position, splits=None, split_number=1):
        """Generates a deteministic train-(test/valid) split. Both splits must contain the same classes and all classes in
        the entire datasets. If no such split can be sampled in 7 passes, returns None.

        :param X: torch tensor, feature values
        :param y: torch tensor, class values
        :param bptt: Number of samples in train + test
        :param eval_position: Number of samples in train, i.e. from which index values are in test
        :param split_number: The split id
        :return:
        """
        done, seed = False, 13

        if splits is None:
            torch.manual_seed(split_number)
            perm = (
                torch.randperm(self.x.shape[0])
                if split_number > 1
                else torch.arange(0, self.x.shape[0])
            )
            ds_shuffled = self[perm]

            while not done:
                if seed > 20:
                    return (
                        None,
                        None,
                    )  # No split could be generated in 7 passes, return None
                random.seed(seed)
                i = (
                    random.randint(0, len(ds_shuffled.x) - bptt)
                    if len(ds_shuffled.x) - bptt > 0
                    else 0
                )
                y_ = ds_shuffled.y[i : i + bptt]

                if self.task_type == "multiclass":
                    # Checks if all classes from dataset are contained and classes in train and test are equal (contain same
                    # classes) and
                    done = len(torch.unique(y_)) == len(torch.unique(ds_shuffled.y))
                    done = done and torch.all(
                        torch.unique(y_) == torch.unique(ds_shuffled.y)
                    )
                    done = done and len(torch.unique(y_[:eval_position])) == len(
                        torch.unique(y_[eval_position:])
                    )
                    done = done and torch.all(
                        torch.unique(y_[:eval_position])
                        == torch.unique(y_[eval_position:])
                    )
                    seed = seed + 1
                else:
                    done = True

            if self.task_type == "multiclass":
                ds_shuffled.y = (
                    (ds_shuffled.y.unsqueeze(-1) > torch.unique(ds_shuffled.y))
                    .sum(axis=1)
                    .unsqueeze(-1)
                )
            ds_shuffled.y = ds_shuffled.y.reshape(self.x.shape[0])

            train_ds = ds_shuffled[i : i + bptt][:eval_position]
            test_ds = ds_shuffled[i : i + bptt][eval_position:]
        else:
            train_inds, test_inds = splits[split_number][0], splits[split_number][1]

            train_ds = self[train_inds]
            test_ds = self[test_inds]

        return train_ds, test_ds

    def __repr__(self):
        return f"{self.name}"


def get_openml_dataset(did, max_samples, shuffled=True):
    """
    :param did: The dataset id
    :param max_samples: The maximum number of samples to return
    :param shuffled: Whether to shuffle the data
    """
    dataset = openml.datasets.get_dataset(did)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="array", target=dataset.default_target_attribute
    )

    assert isinstance(X, np.ndarray) and isinstance(
        y, np.ndarray
    ), "Not a NP Array, skipping"

    np.random.seed(13)
    order = np.arange(y.shape[0])
    if shuffled:
        np.random.shuffle(order)
    X, y = torch.tensor(X[order]), torch.tensor(y[order])
    X, y = X[:max_samples], y[:max_samples] if max_samples else (X, y)

    return X, y, list(np.where(categorical_indicator)[0]), attribute_names


def cap_dataset(
    X,
    y,
    categorical_feats,
    is_regression,
    num_feats,
    max_num_classes,
    min_samples,
    max_samples,
    return_capped=False,
):
    modifications = {
        "samples_capped": False,
        "classes_capped": False,
        "feats_capped": False,
    }

    if X.shape[1] > num_feats:
        if return_capped:
            X = X[:, 0:num_feats]
            categorical_feats = [c for c in categorical_feats if c < num_feats]
            modifications["feats_capped"] = True
        else:
            raise ValueError(f"Too many features! ({X.shape} vs {num_feats}), skipping")
    elif X.shape[1] == 0:
        raise ValueError(f"No features! ({X.shape}), skipping")
    if X.shape[0] >= max_samples + 1:
        if return_capped:
            modifications["samples_capped"] = True
        else:
            raise ValueError("Too many samples")

    if X.shape[0] < min_samples:
        raise ValueError(f"Too few samples left")

    if not is_regression and len(np.unique(y)) > max_num_classes:
        if return_capped:
            X = X[y < np.unique(y)[10]]
            y = y[y < np.unique(y)[10]]
            modifications["classes_capped"] = True
        else:
            raise ValueError(f"Too many classes")

    return X, y, categorical_feats, modifications


import os


def load_openml_list(
    dids,
    filter_for_nan=False,
    num_feats=100,
    min_samples=100,
    max_samples=400,
    multiclass=True,
    max_num_classes=10,
    shuffled=True,
    return_capped=False,
    load_data=True,
    return_as_lists=True,
):
    datasets = []
    tids, dids = zip(
        *[did.split("@") if type(did) == str else (None, did) for did in dids]
    )
    # if file does not exist write header
    path = f"{np.sum(np.array(dids).astype(float))}.csv"
    if not os.path.isfile(path):
        openml_list = openml.datasets.list_datasets(dids)
        datalist = pd.DataFrame.from_dict(openml_list, orient="index")
        datalist.reset_index(drop=True).to_csv(path, index=False)
    else:
        datalist = pd.read_csv(path)
        datalist = datalist.set_index("did", drop=False)
        datalist.index.name = None

    print(f"Number of datasets: {len(datalist)}")

    if filter_for_nan:
        datalist = datalist[datalist["NumberOfInstancesWithMissingValues"] == 0]
        print(
            f"Number of datasets after Nan and feature number filtering: {len(datalist)}"
        )

    if not load_data:
        return None, datalist

    for i in range(len(dids)):
        entry = datalist.loc[int(dids[i])]
        print("Loading", entry["name"], entry.did, "..")
        if not return_capped and (
            entry.NumberOfInstances > max_samples or entry.NumberOfFeatures > num_feats
        ):
            print("Skipping: too many features or samples")
            continue
        splits = None
        if tids[i] is not None:  # If task id is provided
            splits = []
            for fold in range(0, 10):
                task = openml.tasks.get_task(tids[i])
                train_indices, test_indices = task.get_train_test_split_indices(
                    repeat=0,
                    fold=fold,
                    sample=0,
                )
                splits += [(train_indices, test_indices)]

        is_regression = entry["NumberOfClasses"] == 0.0 or np.isnan(
            entry["NumberOfClasses"]
        )
        try:
            X, y, categorical_feats, attribute_names = get_openml_dataset(
                int(entry.did), max_samples + 1, shuffled=shuffled
            )
            X, y, categorical_feats, modifications = cap_dataset(
                X,
                y,
                categorical_feats,
                is_regression,
                num_feats,
                max_num_classes,
                min_samples,
                max_samples,
                return_capped=return_capped,
            )
        except Exception as e:
            print(e)
            continue

        if return_as_lists:
            datasets += [
                [
                    entry["name"],
                    X,
                    y,
                    categorical_feats,
                    attribute_names,
                    modifications,
                    splits,
                ]
            ]
        else:
            modifications = DatasetModifications(**modifications)
            datasets += [
                TabularDataset(
                    name=entry["name"],
                    x=X,
                    y=y,
                    categorical_feats=categorical_feats,
                    attribute_names=attribute_names,
                    modifications=modifications,
                    splits=splits,
                    task_type="regression" if is_regression else "multiclass",
                )
            ]

    return datasets, datalist


def get_benchmark_dids_for_task(task_type, split="train"):
    if task_type == "multiclass":
        if split == "test":
            return open_cc_dids_classification
        elif split == "valid":
            return valid_dids_classification
        elif split == "debug":
            return open_cc_dids_classification[:1]
        else:
            raise ValueError("Unknown split")
    elif task_type == "regression":
        if split == "test":
            return automl_dids_regression
        elif split == "valid":
            return valid_dids_regression
        elif split == "debug":
            return automl_dids_regression[:1]
        else:
            raise ValueError("Unknown split")
    else:
        raise ValueError("Unknown task type.")


def get_benchmark_for_task(
    task_type,
    split="test",
    max_samples=2000,
    max_features=100,
    max_classes=10,
    filter_for_nan=False,
    return_capped=False,
    return_as_lists=True,
    n_max=50,
):
    if task_type == "regression" or task_type == "multiclass":
        dids = get_benchmark_dids_for_task(task_type, split)
        print(dids)
        openml_datasets, openml_datasets_df = load_openml_list(
            dids,
            multiclass=True,
            shuffled=True,
            filter_for_nan=filter_for_nan,
            max_samples=max_samples,
            num_feats=max_features,
            return_capped=return_capped,
            max_num_classes=max_classes,
            return_as_lists=return_as_lists,
        )
        openml_datasets, openml_datasets_df = (
            openml_datasets[:n_max],
            openml_datasets_df[:n_max],
        )
    elif task_type == "survival":
        openml_datasets, openml_datasets_df = [
            get_dd(),
            get_rossi(),
            get_waltons(),
            get_support(),
            get_aids(),
            get_gbsg2(),
            get_breast_cancer(),
            get_flchain(),
            get_veterans_lung_cancer(),
            get_whas500(),
        ], None
    else:
        raise NotImplementedError(f"Unknown task type {task_type}")

    return openml_datasets, openml_datasets_df


class SurvivalDataset(TabularDataset):
    def __init__(self, event_observed: torch.tensor, y: torch.tensor, **kwargs):
        """ """
        self.task_type = "survival"
        if "task_type" in kwargs:
            del kwargs["task_type"]
        super().__init__(task_type=self.task_type, y=y, **kwargs)
        self.event_observed = (
            event_observed if event_observed is not None else torch.ones_like(y)
        )
        self.x = SurvivalDataset.append_censoring_to_x(self.x, event_observed)

    @staticmethod
    def get_missing_event_indicator():
        return -1

    @staticmethod
    def append_censoring_to_x(x, event_observed):
        if event_observed is None:
            event_observed = torch.ones_like(x[:, 0]).float()
            event_observed[:] = SurvivalDataset.get_missing_event_indicator()
        assert torch.all(
            torch.logical_or(
                torch.logical_or(event_observed == 0, event_observed == 1),
                event_observed == -1,
            )
        ), f"Censoring must be -1, 0 or 1 has {event_observed.unique()}"
        return torch.cat([event_observed.unsqueeze(-1), x], -1)

    @staticmethod
    def extract_censoring_from_x(x):
        # returns (X, Censoring)
        return x[:, 1:], x[:, 0]

    def event(self):
        return self.x[:, 0:1]

    # @property
    # def x(self):
    #    return torch.cat([self.x_data, self.censoring.unsqueeze(-1)], -1)

    def __repr__(self):
        return f"{self.name}"


def get_waltons():
    from lifelines.datasets import load_waltons

    df = load_waltons()  # returns a Pandas DataFrame
    return get_lifelines_survival("waltons", df, "T", "E")


def get_rossi():
    from lifelines.datasets import load_rossi

    df = load_rossi()  # returns a Pandas DataFrame
    return get_lifelines_survival("rossi", df, "week", "arrest")


def get_dd():
    from lifelines.datasets import load_dd

    df = load_dd()  # returns a Pandas DataFrame
    return get_lifelines_survival("dd", df, "duration", "observed")


def get_lifelines_survival(name, df, t, e, shuffled=True):
    cat_columns = df.select_dtypes(["object", "category"]).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.astype("category"))
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)

    df = df.sample(frac=1) if shuffled else df
    df = df
    T = df[t]
    E = df[e]
    df = df.drop(columns=[t, e])

    return SurvivalDataset(
        x=torch.tensor(df.values).float(),
        y=torch.tensor(T).float(),
        name=name,
        attribute_names=list(df.columns),
        event_observed=torch.tensor(E).float(),
    )


def get_support():
    # Install
    # git clone https://github.com/autonlab/auton-survival.git
    # mv auton-survival/auton_survival/ auton_survival
    # rm auton-survival/ -rf
    from auton_survival import datasets as auton_datasets

    outcomes, features = auton_datasets.load_dataset("SUPPORT")
    features = features.assign(event=outcomes.event, time=outcomes.time)
    return get_lifelines_survival("SUPPORT", features, "time", "event", shuffled=True)


def get_flchain():
    # !pip install -U scikit-learn==0.21.3 --no-deps
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_flchain()  # returns a Pandas DataFrame
    return get_scikit_survival("flchain", df)


def get_gbsg2():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_gbsg2()  # returns a Pandas DataFrame
    return get_scikit_survival("gbsg2", df)


def get_whas500():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_whas500()  # returns a Pandas DataFrame
    return get_scikit_survival("whas500", df)


def get_veterans_lung_cancer():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_veterans_lung_cancer()  # returns a Pandas DataFrame
    return get_scikit_survival("veterans_lung_cancer", df)


def get_breast_cancer():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_breast_cancer()  # returns a Pandas DataFrame
    return get_scikit_survival("breast_cancer", df)


def get_aids():
    from sksurv import datasets as sksurv_datasets

    df = sksurv_datasets.load_aids()  # returns a Pandas DataFrame
    return get_scikit_survival("aids", df)


def get_scikit_survival(name, sksurv_ds, shuffled=True):
    #!pip install scikit-survival==0.10.0 --no-deps
    event, time = list(zip(*sksurv_ds[1].tolist()))
    sksurv_ds = sksurv_ds[0].assign(event=event, time=time)
    return get_lifelines_survival(name, sksurv_ds, "time", "event", shuffled=shuffled)


# OpenML CC-18 Benchmark (Filtered by N_samples < 2000, N feats < 100, N classes < 10)
open_cc_dids_classification = [
    11,
    14,
    15,
    16,
    18,
    22,
    23,
    29,
    31,
    37,
    50,
    54,
    188,
    458,
    469,
    1049,
    1050,
    1063,
    1068,
    1510,
    1494,
    1480,
    1462,
    1464,
    6332,
    23381,
    40966,
    40982,
    40994,
    40975,
]

# Grinzstjan Benchmark
grinzstjan_numerical_regression = [
    "361072@44132",
    "361073@44133",
    "361074@44134",
    "361075@44135",
    "361076@44136",
    "361077@44137",
    "361078@44138",
    "361079@44139",
    "361080@44140",
    "361081@44141",
    "361082@44142",
    "361083@44143",
    "361084@44144",
    "361085@44145",
    "361086@44146",
    "361087@44147",
    "361088@44148",
    "361089@44025",
    "361090@44026",
    "361091@44027",
]
grinzstjan_categorical_regression = [
    "361093@44055",
    "361094@44056",
    "361096@44059",
    "361097@44061",
    "361098@44062",
    "361099@44063",
    "361101@44065",
    "361102@44066",
    "361103@44068",
    "361104@44069",
    "361287@45041",
    "361288@45042",
    "361289@45043",
    "361291@45045",
    "361292@45046",
    "361293@45047",
    "361294@45048",
]
grinzstjan_numerical_classification = [
    "361055@44089",
    "361060@44120",
    "361061@44121",
    "361062@44122",
    "361063@44123",
    "361065@44125",
    "361066@44126",
    "361068@44128",
    "361069@44129",
    "361070@44130",
    "361273@45022",
    "361274@45021",
    "361275@45020",
    "361276@45019",
    "361277@45028",
    "361278@45026",
]
grinzstjan_categorical_classification = [
    "361110@44156",
    "361111@44157",
    "361113@44159",
    "361114@44160",
    "361115@44161",
    "361116@44162",
    "361127@44186",
]

# Automl Benchmark
automl_dids_classification = [
    "2073@181",
    "3945@1111",
    "7593@1596",
    "10090@1457",
    "146818@40981",
    "146820@40983",
    "167120@23517",
    "168350@1489",
    "168757@31",
    "168784@40982",
    "168868@41138",
    "168909@41163",
    "168910@41164",
    "168911@41143",
    "189354@1169",
    "189355@41167",
    "189356@41147",
    "189922@41158",
    "190137@1487",
    "190146@54",
    "190392@41144",
    "190410@41145",
    "190411@41156",
    "190412@41157",
    "211979@41168",
    "211986@4541",
    "359953@1515",
    "359954@188",
    "359955@1464",
    "359956@1494",
    "359957@1468",
    "359958@1049",
    "359959@23",
    "359960@40975",
    "359961@12",
    "359962@1067",
    "359963@40984",
    "359964@40670",
    "359965@3",
    "359966@40978",
    "359967@4134",
    "359968@40701",
    "359969@1475",
    "359970@4538",
    "359971@4534",
    "359972@41146",
    "359973@41142",
    "359974@40498",
    "359975@40900",
    "359976@40996",
    "359977@40668",
    "359979@4135",
    "359980@1486",
    "359981@41027",
    "359982@1461",
    "359983@1590",
    "359984@41169",
    "359985@41166",
    "359986@41165",
    "359987@40685",
    "359988@41159",
    "359989@41161",
    "359990@41150",
    "359991@41162",
    "359992@42733",
    "359993@42734",
    "359994@42732",
    "360112@42746",
    "360113@42742",
    "360114@42769",
    "360975@43072",
]
automl_dids_regression = [
    "167210@41021",
    "233211@42225",
    "233212@42571",
    "233213@4549",
    "233214@42572",
    "233215@42570",
    "317614@42705",
    "359929@42728",
    "359930@550",
    "359931@546",
    "359932@541",
    "359933@507",
    "359934@505",
    "359935@287",
    "359936@216",
    "359937@41540",
    "359938@42688",
    "359939@422",
    "359940@416",
    "359941@42724",
    "359942@42727",
    "359943@42729",
    "359944@42726",
    "359945@42730",
    "359946@201",
    "359948@41980",
    "359949@42731",
    "359950@531",
    "359951@42563",
    "359952@574",
    "360932@3050",
    "360933@3277",
    "360945@43071",
]

# Validation datasets
valid_dids_regression = [
    509,
    695,
    686,
    197,
    690,
    670,
    689,
    43403,
    519,
    712,
    43384,
    688,
    41928,
    44026,
    43452,
    195,
    44052,
    41943,
    43878,
    549,
    703,
    42110,
    42368,
    41514,
    213,
    555,
    200,
    676,
    42361,
    301,
    506,
    43927,
    42712,
    229,
    534,
    191,
    42183,
    42370,
    308,
    232,
    566,
    223,
    434,
    520,
    43882,
    23515,
    196,
    43881,
    296,
    511,
    43672,
    43465,
    666,
    1029,
    545,
    207,
    42369,
    512,
    227,
    8,
    42364,
    44028,
    504,
    23516,
    43978,
    224,
    41539,
    41187,
    556,
    567,
    43466,
    42362,
    43919,
    482,
    41265,
    1097,
    456,
    42636,
    206,
    1199,
    299,
    294,
    664,
    230,
    194,
    42360,
    43926,
    560,
    43477,
    1099,
    40916,
    189,
    231,
    298,
    41523,
    43944,
    1027,
    42224,
    561,
    44150,
    540,
    44108,
    522,
    1245,
    203,
    503,
    43093,
    494,
    516,
    42363,
    40601,
    42464,
    42545,
    42367,
    500,
]
valid_dids_regression = list(
    set(valid_dids_regression) - {43093, 43384, 43403, 43452, 43465, 43466, 43477}
)
valid_dids_regression = list(
    np.array(valid_dids_regression)[np.array(valid_dids_regression) < 43093]
)

valid_dids_classification = [
    470,
    25,
    346,
    20,
    61,
    1069,
    23499,
    185,
    1073,
    461,
    1498,
    1511,
    40496,
    1100,
    40474,
    1523,
    444,
    1465,
    182,
    679,
    1001,
    1441,
    333,
    40589,
    1504,
    42793,
    59,
    1443,
    468,
    4,
    481,
    1488,
    30,
    476,
    40707,
    1116,
    40686,
    1499,
    39,
    53,
    40710,
    41939,
    311,
    41082,
    164,
    41972,
    40646,
    1442,
    40497,
    41919,
    1120,
    1057,
    342,
    464,
    443,
    449,
    1448,
    316,
    13,
    41430,
    466,
    451,
    473,
    60,
    48,
    40678,
    1459,
    51,
    682,
    35,
    42532,
    40669,
    446,
    683,
    1516,
    43892,
    467,
    28,
    44,
    34,
    40592,
    734,
    1044,
    40690,
    285,
    42172,
    1512,
    41538,
    1501,
    40,
    1497,
    45060,
    186,
    1563,
    1476,
    56,
    343,
    40706,
    1040,
    40681,
    40682,
    1478,
    40997,
    472,
    337,
    480,
    187,
    55,
    42167,
    981,
    1075,
    1471,
    40693,
    40704,
    1060,
    987,
    1513,
    1520,
    43,
    1564,
    42585,
    1056,
    1479,
    40709,
    40588,
    312,
    1038,
    377,
    460,
    24,
    1055,
    1455,
    1506,
    685,
    1222,
    26,
    4153,
    452,
    448,
    1508,
    1485,
    479,
    465,
    477,
    40665,
    4154,
    44200,
    42169,
    40663,
    41671,
    336,
    338,
    1527,
    1037,
    475,
    62,
    474,
    1064,
    46,
    375,
    310,
    1046,
    1451,
    1121,
    1473,
    9,
    1048,
    40705,
    1447,
    40594,
    450,
    38,
    40910,
    1519,
    275,
    40680,
    1460,
    1053,
    41976,
    463,
    1071,
    1547,
    803,
    1117,
    747,
    488,
    1167,
    45023,
    1054,
    40677,
    1484,
    1463,
    40536,
    4340,
    1061,
    49,
    40711,
    1059,
    1467,
    36,
    42192,
    1477,
    1412,
    694,
    1065,
    1495,
    1496,
    1041,
    340,
    1045,
    453,
    10,
    459,
    276,
    40666,
    1446,
    378,
    327,
    32,
    1450,
    1115,
    329,
    41,
    2,
    1490,
]
valid_dids_regression = [
    41539,
    230,
    298,
    44965,
    42464,
    43477,
    44152,
    42370,
    41969,
    500,
    686,
    43672,
    191,
    41938,
    44990,
    1199,
    195,
    40916,
    664,
    44052,
    296,
    678,
    198,
    44958,
    41928,
    194,
    196,
    42367,
    229,
    1245,
    494,
    551,
    294,
    23515,
    41968,
    222,
    42363,
    41514,
    203,
    41943,
    528,
    676,
    40601,
    497,
    44969,
    43093,
    526,
    44983,
    43452,
    224,
    223,
    1027,
    1029,
    703,
    44793,
    4544,
    512,
    529,
    227,
    42110,
    299,
    43919,
    688,
    712,
    197,
    232,
    42360,
    689,
    544,
    522,
    492,
    43927,
    45075,
    566,
    41523,
    516,
    44028,
    543,
    482,
    663,
    42900,
    44966,
    511,
    670,
    42712,
    42361,
    206,
    405,
    519,
    506,
    301,
    42636,
    695,
    43878,
    42176,
    42368,
    1099,
    43465,
    210,
    199,
    45062,
    44968,
    43384,
    44212,
    1228,
    43466,
    665,
    43978,
    534,
    520,
    231,
    560,
    42183,
    509,
    561,
    523,
    42545,
    200,
    1097,
    41187,
    42437,
    42366,
    308,
    43926,
    567,
    1070,
    44150,
    540,
    213,
    207,
    504,
    43403,
    666,
    189,
    23516,
    209,
    549,
    44973,
    42364,
    503,
    42362,
    456,
    555,
    535,
    579,
    521,
    42369,
    45074,
    434,
    8,
    690,
]
