from typing import Callable, List, Dict

import numpy as np

from src.si.data.dataset import Dataset
from src.si.model_selection.split import train_test_split

def cross_validate(model,
                   dataset: Dataset,
                   scoring: Callable = None,
                   cv: int = 3,
                   test_size: float = 0.3) -> Dict[str, List[float]]:
    """
    It performs Cross validation on a given model and dataset

    Parameters
    ----------
    :param model: model to be evaluated
    :param dataset: Dataser
        dataset to be used
    :param scoring: Callable
        scoring function
    :param cv: int
        number of folds
    :param test_size: float
        test size

    Return
    ------
    :return: Dict[str, List[float]]
        dictionarie of scores
    """
    scores = {
        'seed': [],
        'train': [],
        'test': []
    }

    # score fo each fold
    for i in range(cv):
        # set the random seed
        random_state = np.random.randint(0, 1000)

        # store seed
        scores['seed'].append(random_state)

        # split dataset in train and test
        train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)

        # train (fit) the model
        model.fit(train)

        # calculate the training scores
        if scoring is None:

            # store training score
            scores['train'].append(model.score(train))

            # store test score
            scores['test'].append(model.score(test))

        else:
            # store training score
            scores['train'].append(scoring(train.y, model.score(train)))

            # store test score
            scores['test'].append(scoring(test.y, model.score(test)))

    return scores