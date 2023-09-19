from __future__ import annotations
from typing import Any, Optional
import sys
import math
import numpy as np
import python_ta
import data


class Tree:
    """A recursive tree data structure.

    Representation Invariants:
        - self._root is not None or self._subtrees == []
    """
    root: Optional[Any]
    subtrees: list[Tree]

    def __init__(self, root: Optional[Any], subtrees: list[Tree]) -> None:
        """Initialize a new Tree with the given root value and subtrees.

        If root is None, the tree is empty.

        Preconditions:
            - root is not none or subtrees == []
        """
        self.root = root
        self.subtrees = subtrees

    def is_empty(self) -> bool:
        """Return whether this tree is empty.
        """
        return self.root is None

    def add_subtree(self, item: Any) -> Tree:
        """
        Adds a subtree to the tree with the item as a root, and returns the subtree added
        """
        tree = Tree(item, [])
        self.subtrees += [tree]
        return self.subtrees[-1]


class BinarySearchTree:
    """A binary search tree data structure.

        Representation Invariants:
           - self.root is not None or self.subtrees == []
       """
    root: Optional[Any]
    left: Optional[BinarySearchTree]
    right: Optional[BinarySearchTree]

    def __init__(self, root: Optional[Any]) -> None:
        """Initialize a new BST containing only the given root value.

        If <root> is None, initialize an empty BST.
        """
        if root is None:
            self.root = None
            self.left = None
            self.right = None
        else:
            self.root = root
            self.left = BinarySearchTree(None)
            self.right = BinarySearchTree(None)

    def is_empty(self) -> bool:
        """Return whether this BST is empty.
        """
        return self.root is None

    def items(self) -> list:
        """Return all of the items in the BST in sorted order.

        """
        if self.is_empty():
            return []
        else:
            return self.left.items() + [self.root] + self.right.items()

    def return_subtree_tuples(self) -> list[tuple]:
        """ Returns tuples of subtrees of all subtress in graphs"""
        if self.root is None:
            return []
        else:
            root = self.root
            if not self.left.is_empty():
                left = self.left
            else:
                left = None
            if not self.right.is_empty():
                right = self.right
            else:
                right = None
            left_roots = self.left.return_subtree_tuples()
            right_roots = self.right.return_subtree_tuples()

            tuple_pairs = [(root, left, right)]
            tuple_pairs.extend(left_roots)
            tuple_pairs.extend(right_roots)

            return tuple_pairs

    def __contains__(self, item: Any) -> bool:
        """Return whether 'player' is in this BST.
        """
        if self.is_empty():
            return False
        else:
            if item == self.root['player']:
                return True
            elif item < self.root['player']:
                return self.left.__contains__(item)
            else:
                return self.right.__contains__(item)

    def contains1(self, item: Any) -> bool:
        """Return whether 'player' is in this BST.
        Used for print_player_career_stats
        """
        if self.is_empty():
            return False
        else:
            if item == self.root[0]:
                return True
            elif item < self.root[0]:
                return self.left.contains1(item)
            else:
                return self.right.contains1(item)

    def _str_indented(self, depth: int) -> str:
        """Return an indented string representation of this BST.

        The indentation level is specified by the <depth> parameter.

        Preconditions:
            - depth >= 0
        """
        if self.is_empty():
            return ''
        else:
            return (
                depth * '  ' + 'depth = ' + str(depth) + ' ' + f'{self.root}\n'
                + self.left._str_indented(depth + 1)
                + self.right._str_indented(depth + 1)
            )

    def access_element(self, name: str) -> dict:
        """ Returns element of BST of player stat dictionaries sorted alphabetically
        """
        if name == self.root['player']:
            return self.root
        elif name < self.root['player']:
            return self.left.access_element(name)
        else:
            return self.right.access_element(name)

    def access_element1(self, name: str) -> dict:
        """ Returns element of BST of player stat tuples
        Used for print_player_career_stats
        """
        if name == self.root[0]:
            return self.root
        elif name < self.root[0]:
            return self.left.access_element1(name)
        else:
            return self.right.access_element1(name)


###################################################################################################
# Creating Stats Tree and Recursing through it to get stats
###################################################################################################

def create_tree1() -> Tree:
    """
    Creates a tree object that consists of player stats
    """
    sys.setrecursionlimit(8250)  # Sets recursion limit to length of dataset list(~8250)
    d = data.get_data()
    d = d.to_dict('records')
    tree = Tree({'player': '', 'age': ''}, [])
    listreturn(tree, tree, d, 0)
    return tree


def listreturn(tree: Tree, original_tree: Tree, stats: list[dict], index: int) -> None:
    """Recursive Helper Function for create_tree1()
    """
    if index == len(stats) - 1:
        tree.add_subtree(stats[index])
        return
    elif stats[index + 1]['player'] == stats[index]['player']:
        b = tree.add_subtree(stats[index])
        listreturn(b, original_tree, stats, index + 1)
    else:
        tree.add_subtree(stats[index])
        listreturn(original_tree, original_tree, stats, index + 1)


def two_stat(age1: int, age2: int, stat: str, tree: Tree) -> list[tuple]:
    """
    Returns a specific stat of all players of age1 and age2 for two concecutive seasons
    >>> b = create_tree1()
    >>> c = two_stat(38,39,'stl_per_game',b)
    """
    if tree.root['age'] == age1 or tree.root['age'] == age2:
        averages = []
        b = tree
        while b.subtrees != [] and (b.root['age'] == age1 or b.root['age'] == age2):
            averages += [(b.root['player'], b.root['age'], b.root[stat], b.root['g'], b.root['tm'])]
            b = b.subtrees[0]
        return averages
    else:
        result = []
        for subtree in tree.subtrees:
            result.extend(two_stat(age1, age2, stat, subtree))
        return result


###################################################################################################
# Calculating Average Decline
###################################################################################################

def combine_trades(slist: list[tuple]) -> list[tuple]:
    """
    Changes list from two_stat and averages the per game stats of players that were
    traded(players who have multiple entries in the same season/age)

    Preconditions:
        - slist is generated from two_stat
    """
    old_list = slist
    x = 0
    while x < len(old_list) - 1:
        if old_list[x][4] == 'TOT':
            old_list.pop(x)
            continue
        if old_list[x][1] == old_list[x + 1][1] and old_list[x][0] == old_list[x + 1][0]:
            total = old_list[x][2] * old_list[x][3] + old_list[x + 1][2] * old_list[x + 1][3]
            sumgames = old_list[x][3] + old_list[x + 1][3]
            avg = total / sumgames
            avg = round(avg, 1)
            old_list[x] = (old_list[x][0], old_list[x][1], avg, sumgames, 'TOTAL')
            old_list.pop(x + 1)
            x = x - 1
        x += 1
    return old_list


def find_median_decline(input_list: list[tuple]) -> float:
    """
    Finds the median decline of a singular stat from a list created by combine_trades
    Not useable for Games Played

    Preconditions:
        - input_list is generated from combine_trades

    >>> b = create_tree1()
    >>> c = two_stat(38,39,'fg_per_game',b)
    >>> c = combine_trades(c)
    >>> find_median_decline(c)
    -0.2222222222222223
    """
    stats = []
    for x in range(0, len(input_list) - 1):
        if input_list[x][0] == input_list[x + 1][0]:
            if (input_list[x][2]) != 0 and not math.isnan(input_list[x][2]) and not math.isnan(input_list[x + 1][2]):
                stats += [(input_list[x + 1][2] - input_list[x][2]) / input_list[x][2]]
    stats.sort()
    mid = (len(stats) - 1) // 2
    if len(stats) % 2 == 1:
        return stats[mid]
    else:
        return (stats[mid] + stats[mid + 1]) / 2.0


def find_median_decline_games(input_list: list[tuple]) -> float:
    """
    Finds the median decline of games played from a list created by combine_trades

    Preconditions:
        - input_list is generated from combine_trades

    >>> tree1 = create_tree1()
    >>> twostat = two_stat(38,39,'g',tree1)
    >>> combined = combine_trades(twostat)
    >>> find_median_decline_games(combined)
    -0.09090909090909091
    """
    stats = []
    for x in range(0, len(input_list) - 1):
        if input_list[x][0] == input_list[x + 1][0]:
            if (input_list[x][3]) != 0 and not math.isnan(input_list[x][3]) and not math.isnan(input_list[x + 1][3]):
                stats += [(input_list[x + 1][3] - input_list[x][3]) / input_list[x][3]]
    stats.sort()
    mid = (len(stats) - 1) // 2
    if len(stats) % 2 == 1:
        return stats[mid]
    else:
        return (stats[mid] + stats[mid + 1]) / 2.0


def find_outliers(input_list: list[tuple]) -> list[tuple]:
    """
    Finds outilers who differ from the median by 1.5*IQR from a list taken from combine_trades
    Returns a list of tuples(percentage difference, name, age during first season, age during second season) for
    these players

    Preconditions:
        - input_list is generated from combine_trades

    >>> tree1 = create_tree1()
    >>> twostat = two_stat(38,39,'fg_per_game',tree1)
    >>> combined = combine_trades(twostat)
    >>> find_outliers(combined)
    [(-0.7058823529411765, 'Charles Oakley', 38.0, 39.0), (1.0, 'Charles Jones', 38.0, 39.0)]
    """
    stats = []
    for x in range(0, len(input_list) - 1):
        if input_list[x][0] == input_list[x + 1][0]:
            if (input_list[x][2]) != 0 and not math.isnan(input_list[x][2]) and not math.isnan(input_list[x + 1][2]):
                stats += [((input_list[x + 1][2] - input_list[x][2]) / input_list[x][2], input_list[x][0],
                           input_list[x][1], input_list[x + 1][1])]
    stats.sort()
    q75, q25 = np.percentile([xs[0] for xs in stats], [75, 25])
    iqr = q75 - q25
    outlier_list = []
    for x in stats:
        if x[0] < q25 - 1.5 * iqr or x[0] > q75 + 1.5 * iqr:
            outlier_list += [x]
    return outlier_list


def return_list_declines(input_list: list[tuple]) -> list[tuple]:
    """
    Returns A list of tuples(name, percentage difference, age during first season, age during second season)
    for all players taken from an input_list from combine_trades

    Preconditions:
        - input_list is generated from combine_trades

    >>> tree1 = create_tree1()
    >>> twostat = two_stat(38,39,'fg_per_game',tree1)
    >>> combined = combine_trades(twostat)
    >>> returned_list = return_list_declines
    >>> returned_list is not None
    True
    """
    stats = []
    for x in range(0, len(input_list) - 1):
        if input_list[x][0] == input_list[x + 1][0]:
            if (input_list[x][2]) != 0 and not math.isnan(input_list[x][2]) and not math.isnan(input_list[x + 1][2]):
                stats += [(input_list[x][0], (input_list[x + 1][2] - input_list[x][2]) / input_list[x][2],
                           input_list[x][1], input_list[x + 1][1])]
    stats.sort()
    return stats


###################################################################################################
# Creating and Using Binary Search Trees for Fast Lookup
###################################################################################################

def create_alphabetical_bst(age: int) -> tuple[BinarySearchTree, BinarySearchTree]:
    """
    Creates 2 Binary Search Trees of a specific age sorted alphabetically for fast lookup of a player's season
    First binary search tree is a-m, second binary search tree is n-z and players with accented first letters

    Preconditions:
        - 27 <= age <= 42
    """
    ds = data.get_data()
    ds = ds.drop(ds[ds['age'] != age].index, inplace=False)
    ds.loc[ds['tm'] == 'TOT', 'tm'] = 'A'  # We use A so if a player is traded, their TOT stats are alphabetically first
    ds = ds.sort_values(['player', 'tm'])
    ds = ds.to_dict('records')
    ds = combine_trades_2(ds)
    half_index = [x[0] for x in enumerate(ds) if x[1]['player'].casefold() < 'n'][-1]
    first_half = ds[:half_index + 1]
    second_half = ds[half_index + 1:]
    return sorted_bst(first_half), sorted_bst(second_half)


def combine_trades_2(slist: list[dict]) -> list[dict]:
    """
    Helper Function for create_alphabetical_bst
    Removes players who have multiple entries because of trades from a list of dictionaries of player stats and instead
    makes one entry with their total stats
    """
    old_list = slist
    x = 0
    totals = set()
    while x < len(old_list) - 1:
        if old_list[x]['tm'] == 'A':  # We check if it is A to see if we have to remove copies
            totals.add(old_list[x]['player'])
        else:
            if old_list[x]['player'] in totals:
                old_list.pop(x)
                x = x - 1
        x += 1
    return old_list


def sorted_bst(half: list) -> BinarySearchTree:
    """
    Helper Function for create_alphabetical_bst and create_singular_stat_bst
    Creates a BST recursively by splitting the sorted list at the halfway point
    """
    if not half:
        return BinarySearchTree(None)
    middle = len(half) // 2
    mid = BinarySearchTree(half[middle])
    mid.left = sorted_bst(half[:middle])
    mid.right = sorted_bst(half[middle + 1:])
    return mid


def create_singular_stat_bst_alphab(age1: int, age2: int, stat: str) -> BinarySearchTree:
    """
    Creates Binary Search Tree with percentage decline for load_all_singular_stat_bst(sorted alphabetically)
    """
    tree1 = create_tree1()
    twostat = two_stat(age1, age2, stat, tree1)
    combined = combine_trades(twostat)
    list_declines = return_list_declines(combined)
    return sorted_bst(list_declines)


def create_singular_stat_bst_vis(age1: int, age2: int, stat: str) -> BinarySearchTree:
    """
    Creates Binary Search Tree with percentage decline sorted by that decline in one stat
    that allows for data visualization
    """
    tree1 = create_tree1()
    twostat = two_stat(age1, age2, stat, tree1)
    combined = combine_trades(twostat)
    list_declines = return_list_declines(combined)
    list_declines = [(x[1], x[0], x[2], x[3]) for x in list_declines]
    list_declines.sort()
    return sorted_bst(list_declines)


def load_alphabetical_bsts() -> dict[int:tuple[BinarySearchTree, BinarySearchTree]]:
    """
    Loads all alphabetical bst's for fast access to a player's stats
    Key values in the output dictionary are age values
    """
    alpha = {}
    for x in range(27, 43):
        alpha[x] = (create_alphabetical_bst(x))
    return alpha


def print_player_career_stats(player: str, stats: dict[int:tuple[BinarySearchTree, BinarySearchTree]]) -> None:
    """
    Prints a players career stats line by line using previously loaded binary search trees

    Preconditions:
        - stats is generated from load_alphabetical_bsts
    """
    if player < 'N':
        a_index = 0
    else:
        a_index = 1
    for x in range(27, 43):
        if stats[x][a_index].__contains__(player):
            print(stats[x][a_index].access_element(player))
        else:
            continue


def load_all_singular_stat_bst() -> dict[tuple:BinarySearchTree]:
    """
    Loads singular_stat_bsts for all ages and all players to store all player decline in Binary Search Trees
    *** Requires loading many BST's, so may take a few minutes(takes me approximately 2 minutes to load)
    *** Apologies if this takes very long, I attempted to implement the suggestion of saving pregenerated data to a file
    but I cannot find a way to store BinarySearchTree Instances in a file
    """
    all_stats = {}
    d = data.get_data()
    statlist = list(d.columns)
    statlist = statlist[10:]  # removing non per game statistics like position or season
    for x in range(27, 44):
        for stat in statlist:
            entry = (x, x + 1, stat)
            all_stats[entry] = create_singular_stat_bst_alphab(entry[0], entry[1], entry[2])
    return all_stats


def return_player_career_decline(player: str, stats: dict[tuple:BinarySearchTree]) -> list[tuple]:
    """
    Returns a player's year-over-year decline in a list

    Preconditions:
        -stats must be pregenerated by load_all_singular_stat_bst()
    """
    d = data.get_data()
    statlist = list(d.columns)
    statlist = statlist[10:]  # removing non per game statistics like position or season
    career_stats = []
    for x in range(27, 44):
        for stat in statlist:
            entry = (x, x + 1, stat)
            if stats[entry].contains1(player):
                career_stats += [(entry[2], stats[entry].access_element1(player))]
            else:
                continue
    return career_stats


def print_player_career_decline(player: str, stats: dict[tuple:BinarySearchTree]) -> None:
    """
    Print's a player's year-over-year decline

    Preconditions:
        -stats must be pregenerated by load_all_singular_stat_bst()
    """
    d = data.get_data()
    statlist = list(d.columns)
    statlist = statlist[10:]  # removing non per game statistics like position or season
    for x in range(27, 44):
        for stat in statlist:
            entry = (x, x + 1, stat)
            if stats[entry].contains1(player):
                print(entry[2], stats[entry].access_element1(player))
            else:
                continue


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    python_ta.check_all(config={
        'extra-imports': ['pandas', 'data', 'typing', 'numpy', 'math', 'sys', 'networkx'],
        'allowed-io': ['print_player_career_decline', 'print_player_career_stats'],
        'max-line-length': 120
    })
