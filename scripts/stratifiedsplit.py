import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def StratifiedSplit(data):
    # Dividing into income categories
    data["income_cat"] = np.ceil(data["median_income"] / 1.5)

    # Putting everything above the 5th category as the 5th category
    data["income_cat"].where(data["median_income"] < 5, other=5.0, inplace=True)

    # Initialize StratifiedShuffleSplit with 1 split and 20% test size
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=29)

    # Perform the stratified split based on the "income_cat" column
    for train_index, test_index in split.split(data, data["income_cat"]):
        strat_train_split = data.iloc[train_index]
        strat_test_split = data.iloc[test_index]

    # Copy the train and test sets
    strat_train_set = data.iloc[train_index].copy()
    strat_test_set = data.iloc[test_index].copy()

    # Drop the temporary stratification column from both sets
    for dataset in (strat_train_set, strat_test_set):
        dataset.drop("income_cat", axis=1, inplace=True)

    # Return the stratified train and test sets
    return strat_train_set, strat_test_set
