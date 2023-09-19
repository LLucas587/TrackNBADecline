import python_ta
import display
import compute

if __name__ == "__main__":
    #########################
    # This block of code will plot median decline for all stats
    #########################
    display.plot_all_avg_decline()

    #########################
    # This block of code will plot median decline for one stat. In its current state, it plots it
    # for 'trb_per_game'(Total rebounds per game)
    # One can modify the stat plotted in the function below
    #########################
    # display.plot_avg_decline('trb_per_game')

    #########################
    # This block of code will create an interactive binary search tree of all players age(30-31) sorted by trb_per_game
    # One can modify the paramaters of compute.create_singular_stat_bst and display.plot_igraph to change the output
    #########################
    # computed = compute.create_singular_stat_bst_vis(30, 31, 'trb_per_game')
    # g = display.man_igraph(computed)
    # display.plot_igraph(g, 'trb_per_game')

    #########################
    # This block of code will print a player's career stats. Currently, it is set to LeBron James
    # One can modify the name of the player below
    #########################
    # a = compute.load_alphabetical_bsts()
    # compute.print_player_career_stats('LeBron James', a)

    #########################
    # This block of code will find outliers who do not fit within the average decline of player's for that stat and age
    # One can modify the paramaters of two_stat(age1,age2,stat) to change the output
    # Output comes in tuples(Percentage Change, Name, Age during first season, Age during second season)
    #########################
    # tree1 = compute.create_tree1()
    # twostat = compute.two_stat(38, 39, 'fg_per_game', tree1)
    # combined = compute.combine_trades(twostat)
    # print(compute.find_outliers(combined))

    #########################
    # This block of code will use binary search trees to allow for fast lookup and plotting of a player's career decline
    # *** Creating these Binary Search Trees may take a few minutes(Takes me approximately 2-3 minutes on an old laptop)
    # The second line plots all stats for a singular player (In this case Lebron James)
    # The third line plots one stat for a singular player (In this case Lebron James an ast_per_game)
    # One can modify the name paramater of the second line and the name/stat paramater of the third line
    #########################
    # bsts = compute.load_all_singular_stat_bst()
    # display.plot_single_decline_allstats('LeBron James', bsts)
    # display.plot_single_decline('LeBron James', 'ast_per_game', bsts)

    if __name__ == "__main__":
        python_ta.check_all(config={
            'extra-imports': ['compute', 'display'],  # the names (strs) of imported modules
            'allowed-io': [],  # the names (strs) of functions that call print/open/input
            'max-line-length': 120,
            'disable': ['W0611']
        })
