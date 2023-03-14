import numpy as np

def get_tokens_to_remove_five(start_pred, impacts_sorted, percentage, num_round):
    if len(impacts_sorted) >= 5:
        combination = {'firts1': [[0]], 'first2': [[0, 1]], 'first5': [[0, 1, 2, 3, 4]]}
    else:
        combination = {'firts1': [[0]]}

    tokens_to_remove = get_tokens_to_change_class(start_pred, impacts_sorted)
    combination['change_class'] = [tokens_to_remove]
    lent = len(impacts_sorted)
    ntokens = int(lent * percentage)
    np.random.seed(0)
    combination['random'] = [np.random.choice(lent, ntokens, ) for _ in range(num_round)]
    return combination


def get_tokens_to_change_class(start_pred, impacts_sorted, delta: float = 0.0):
    tokens_to_remove = []
    positive_match = start_pred > .5
    # delta = -delta if not positive else delta
    index = np.arange(0, len(impacts_sorted))

    if not positive_match:
        index = index[::-1]  # start removing negative impacts to push the score towards match if not positive

    delta_score_to_achieve = abs(start_pred - 0.5) + delta
    current_delta_score = 0

    for i in index:
        current_token_impact = impacts_sorted[i] * (1 if positive_match else -1)
        # remove positive impact if element is match, neg impacts if no match
        if current_token_impact > 0:
            tokens_to_remove.append(i)
            current_delta_score += current_token_impact
        else:  # there are no more tokens with positive (negative) impacts
            break

        # expected_delta = np.abs(np.sum(impacts_sorted[tokens_to_remove]))

        if current_delta_score >= delta_score_to_achieve:
            break

    return tokens_to_remove


def get_tokens_to_remove_aopc(impacts_sorted, num_round, k=10):
    min_tokens = min(len(impacts_sorted), k)
    combination = {f'MoRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)}
    np.random.seed(0)
    lent = len(impacts_sorted)
    for turn_n_tokens in range(1, min_tokens + 1):
        combination[f'random_{turn_n_tokens}'] = [np.random.choice(lent, turn_n_tokens, replace=False) for _ in
                                                  range(num_round)]
    return combination


def get_tokens_to_remove_degradation(start_pred, impacts_sorted, num_round, random=False):
    lent = len(impacts_sorted)
    min_tokens = lent
    if start_pred > .5:
        combination = {f'MoRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)}
        combination.update(**{f'LeRF_{i}': [np.arange(lent - i, lent)] for i in range(1, min_tokens + 1)})
    else:
        combination = {f'MoRF_{i}': [np.arange(lent - i, lent)] for i in range(1, min_tokens + 1)}
        combination.update(**{f'LeRF_{i}': [np.arange(i)] for i in range(1, min_tokens + 1)})
    np.random.seed(0)
    if random is True:
        for turn_n_tokens in range(1, min_tokens + 1):
            combination[f'random_{turn_n_tokens}'] = [np.random.choice(lent, turn_n_tokens, replace=False) for _ in
                                                      range(num_round)]
    return combination


def get_tokens_to_remove_sufficiency(start_pred, impacts_sorted, num_round, k=10):
    lent = len(impacts_sorted)
    min_tokens = min(lent, k)
    if start_pred > .5:
        combination = {f'top_{i}': [np.arange(i, lent)] for i in range(1, min_tokens + 1)}
    else:
        combination = {f'top_{i}': [np.arange(lent - i)] for i in range(1, min_tokens + 1)}
    np.random.seed(0)
    for turn_n_tokens in range(1, min_tokens + 1):
        combination[f'random_{turn_n_tokens}'] = [
            np.setdiff1d(np.arange(lent), np.random.choice(lent, turn_n_tokens, replace=False)) for _ in
            range(num_round)]
    return combination


def get_tokens_to_remove_single_units(impacts_sorted):
    lent = len(impacts_sorted)
    combination = {f'unit_{i}': [[i]] for i in range(lent)}
    return combination


def get_tokens_to_remove_incrementally(impacts_sorted, limit=10):
    limit = min(len(impacts_sorted), limit + 1)  # + 1 because range excludes the right extreme
    return {f'incremental_{i}': [range(i)] for i in range(1, limit)}

def get_tokens_from_neg_to_pos_and_back(start_pred, impacts_sorted, delta: float=0.0):
    tokens_to_remove = list()
    index = np.arange(len(impacts_sorted), -1, -1)  # start removing negative impacts to push the score towards match

    delta_score_to_achieve = abs(start_pred - 0.5) + delta
    current_delta_score = 0

    for i in index:
        current_token_impact = -impacts_sorted[i]
        # remove positive impact if element is match, neg impacts if no match
        if current_token_impact > 0:
            tokens_to_remove.append(i)
            current_delta_score += current_token_impact
        else:  # there are no more tokens with positive (negative) impacts
            break

        if current_delta_score >= delta_score_to_achieve:
            break

    # TODO: Rimuoviamo token finché non diventa positivo poi aggiungiamo per farlo tornare negativo e vediamo quali DU
    #  spaiate (token) sono più utili
    #  Per fare questo è richiesto di rivalutare il sistema prima? I think so?

    return tokens_to_remove


def swap_tokens_to_match(start_pred, impacts_sorted, delta: float=0.0):
    # TODO: to complete
    index = np.arange(len(impacts_sorted), -1, -1)

    delta_score_to_achieve = abs(start_pred - 0.5) + delta
    current_delta_score = 0

    for i in index:
        current_token_impact = -impacts_sorted[i]

        if current_token_impact > 0:
            pass