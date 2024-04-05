from typing import Sequence


def group_data(data_dict: dict[str, Sequence]) -> list[tuple]:
    """Turns a dict of results into a list of experiences."""
    data_seqs = [data_dict[key] for key in data_dict]
    data_list = [data for data in zip(*data_seqs)]
    return data_list
