from sympy import Rational

from applpy import MarkovChain


def _discrete_time_markov_chain_from_dtmc_notebook():
    return MarkovChain(
        [
            [Rational(3, 10), Rational(3, 10), Rational(4, 10), 0, 0],
            [0, Rational(3, 10), Rational(3, 10), Rational(4, 10), 0],
            [0, 0, Rational(3, 10), Rational(3, 10), Rational(4, 10)],
            [Rational(3, 10), Rational(3, 10), Rational(4, 10), 0, 0],
            [Rational(3, 10), Rational(3, 10), Rational(4, 10), 0, 0],
        ],
        states=["blue", "green", "black", "yellow", "orange"],
    )


def test_discrete_time_markov_chain_long_run_probabilities_from_dtmc_notebook():
    discrete_time_markov_chain = _discrete_time_markov_chain_from_dtmc_notebook()
    long_run_probabilities = discrete_time_markov_chain.long_run_probs(method="rational").tolist()
    expected_row = [Rational(147, 1070), Rational(21, 107), Rational(37, 107), Rational(39, 214), Rational(74, 535)]

    assert len(long_run_probabilities) == 5
    assert all(row == expected_row for row in long_run_probabilities)


def test_discrete_time_markov_chain_path_probabilities_from_dtmc_notebook():
    discrete_time_markov_chain = MarkovChain(
        P=[[0.97, 0.03], [0.04, 0.96]],
        init=[0.7, 0.3],
        states=["red", "blue"],
    )

    assert discrete_time_markov_chain.probability([(2, "blue"), (1, "red"), (0, "red")]) == 0.020369999999999996
    assert discrete_time_markov_chain.probability([(1, "blue")]) == 0.309
    assert discrete_time_markov_chain.probability([(2, "blue")], given=[(1, "blue")]) == 0.96


def test_discrete_time_markov_chain_state_classification_from_dtmc2_notebook():
    discrete_time_markov_chain = MarkovChain(
        P=[
            [0, 0.1, 0.5, 0, 0.2, 0.2, 0, 0],
            [0.1, 0.2, 0, 0, 0.6, 0, 0, 0.1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.2, 0, 0.8],
            [0, 0.3, 0, 0, 0, 0.4, 0.1, 0.2],
            [0, 0, 0, 0, 0, 0.6, 0, 0.4],
        ],
        states=["blue", "green", "purple", "red", "yellow", "black", "orange", "white"],
    )

    assert discrete_time_markov_chain.classify_states() == {
        "Transient": ["blue", "green", "orange"],
        "Recurrent 1": ["red", "yellow"],
        "Recurrent 2": ["black", "white"],
        "Recurrent 3": ["purple"],
    }
    assert discrete_time_markov_chain.reducible is True


def test_discrete_time_markov_chain_reachability_from_dtmc2_notebook():
    discrete_time_markov_chain = MarkovChain(
        P=[
            [0, 0.1, 0.5, 0, 0.2, 0.2, 0, 0],
            [0.1, 0.2, 0, 0, 0.6, 0, 0, 0.1],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.2, 0, 0.8],
            [0, 0.3, 0, 0, 0, 0.4, 0.1, 0.2],
            [0, 0, 0, 0, 0, 0.6, 0, 0.4],
        ],
    )

    reachability = discrete_time_markov_chain.reachability()
    assert reachability.shape == (8, 8)
    assert bool(reachability[0][2]) is True
    assert bool(reachability[0][6]) is False
    assert bool(reachability[3][4]) is True
    assert bool(reachability[5][0]) is False
    assert bool(reachability[6][7]) is True


def test_discrete_time_markov_chain_absorption_from_dtmc3_notebook():
    transition_matrix = [
        [1, 0, 0, 0],
        [Rational(3, 10), Rational(4, 10), Rational(1, 10), Rational(2, 10)],
        [0, 0, 1, 0],
        [Rational(2, 10), Rational(3, 10), Rational(4, 10), Rational(1, 10)],
    ]
    discrete_time_markov_chain = MarkovChain(transition_matrix, states=["red", "blue", "black", "green"])

    assert discrete_time_markov_chain.absorption_prob("red") == [1, Rational(31, 48), 0, Rational(7, 16)]
    assert discrete_time_markov_chain.absorption_steps() == [0, Rational(55, 24), 0, Rational(15, 8)]


def test_discrete_time_markov_chain_long_run_probabilities_from_dtmc3_notebook():
    transition_matrix = [
        [Rational(2, 10), 0, 0, Rational(8, 10)],
        [0, 1, 0, 0],
        [Rational(1, 10), Rational(4, 10), Rational(2, 10), Rational(3, 10)],
        [Rational(7, 10), 0, 0, Rational(3, 10)],
    ]
    discrete_time_markov_chain = MarkovChain(transition_matrix, states=["red", "blue", "black", "green"])

    assert discrete_time_markov_chain.long_run_probs(method="rational").tolist() == [
        [Rational(7, 15), 0, 0, Rational(8, 15)],
        [0, 1, 0, 0],
        [Rational(7, 30), Rational(1, 2), 0, Rational(4, 15)],
        [Rational(7, 15), 0, 0, Rational(8, 15)],
    ]
