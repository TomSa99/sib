from typing import Callable, List, Dict, Tuple, Any

import numpy as np

from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import cross_validate


def randomized_search_cv(model,
                         dataset: Dataset,
                         parameter_distribution: Dict[str, Tuple],
                         scoring: Callable = None,
                         cv: int = 3,
                         n_iter: int = 10,
                         test_size: float = 0.3) -> dict[str, list[Any]]:
    """
    Performs a Randomized search cross validation on a given model and dataset

    Parameters
    ----------
    :param model: model to be evaluated
    :param dataset: Dataset
        dataset to be used
    :param parameter_distribution: Dict[str, Tuple]
        parameteres for the search
    :param scoring: Callable
        scoring function
    :param cv: int
        number of folds
    :param n_iter: int
        number of random parameter combinations
    :param test_size: float
        size of the test dataset

    Return
    ------
    :return: List[Dict[str, Any]]
        A list of dictionaries with the combination of parameters and training and test scores
    """

    scores = {'parameters': [],
              'seeds': [],
              'train': [],
              'test': []}

    # check the parameter grid
    for parameter in parameter_distribution:
        # checks if the parameter exists in the model (True or False)
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}")

    # set n_iter parameter combinations
    for i in range(n_iter):

        # set random seed
        random_state = np.random.randint(0, 1000)

        # store the seeed
        scores['seeds'].append(random_state)

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in parameter_distribution.items():
            # set the combination of parameters and its values to the model
            setattr(model, parameter, np.random.choice(value))

        # set the parameters to the model
        for parameter, value in parameter_distribution.items():
            setattr(model, parameter, value)

        # perform cross validation with the combination of parameters
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # store the parameters combinations and scores
        scores['parameters'].append(parameters)
        scores['train'].append(score['train'])
        scores['test'].append(score['test'])

    return scores