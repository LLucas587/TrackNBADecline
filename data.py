import pandas as pd
import python_ta


def get_data() -> pd.DataFrame:
    """
    Function that takes data from the player per game dataset and removes rows from the dataset of players under the age
    of 27 and of seasons before 1994, before sorting that data by name and age.
    It also removes some columns of statistics that will not be tracked.
    """
    b = pd.read_csv('nba data set/Player Per Game.csv')
    b = b.drop(b[b['age'] < 27].index, inplace=False)
    b = b.drop(b[b['season'] < 1994].index, inplace=False)
    b = b.drop(b[b['lg'] != 'NBA'].index, inplace=False)
    b = b.drop(columns=['birth_year', 'seas_id'])
    b = b.sort_values(['player', 'age'])

    return b


if __name__ == "__main__":
    python_ta.check_all(config={
        'extra-imports': ['pandas'],  # the names (strs) of imported modules
        'allowed-io': [],  # the names (strs) of functions that call print/open/input
        'max-line-length': 120
    })
