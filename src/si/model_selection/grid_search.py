import itertools

from typing import Callable, List, Dict, Tuple, Any

from src.si.data.dataset import Dataset
from src.si.model_selection.cross_validate import cross_validate


def grid_search_cv(model,
                   dataset: Dataset,
                   parameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.3) -> List[Dict[str, Any]]:
    """
    Performs a Grid search cross validation on a given model and dataset

    Parameters
    ----------
    :param model: model to be evaluated
    :param dataset: Dataser
        dataset to be used
    :param parameter_grid: dict
        parameter grid
    :param scoring: Callable
        scoring function
    :param cv: int
        number of folds
    :param test_size: float
        test size

    Return
    ------
    :return: List[Dict[str, Any]]
        dictionarie of scores
    """
    # check the parameter grid
    for parameter in parameter_grid:
        # checks if the parameter exists in the model (True or False)
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}")

    scores = []

    # calculates the cartesian product for each combination of parameter
    for combination in itertools.product(*parameter_grid.values()):

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            # set the combination of parameters and its values to the model
            setattr(model, parameter, value)
            # store the parameter and its value
            parameters[parameter] = value

        # calculate the model score
        score = cross_validate(model=model, dataset=dataset, scoring=scoring, cv=cv, test_size=test_size)

        # store the parameter combination and the score
        score['parameters'] = parameters

        # add the scores
        scores.append(score)

    return scores
