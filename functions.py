import matplotlib.pyplot as plt   # Visualization

###[RQ3]

def plot_posts_time(posts_time):
    '''
    Plot the daily temporal distribution of posts published
    '''
    f = plt.figure()
    plt.xticks(range(0,25),["{}:00".format(x).zfill(5) for x in range(0,25)], rotation = 45)
    plt.ylabel("# posts published", fontsize=14, labelpad=20)
    plt.xlabel("Time of the day", fontsize=14, labelpad=20)
    plt.title("Daily temporal distribution of posts published", fontsize=18)
    plt.hist(posts_time, bins=range(25), color='#00ff00', ec="k")
    f.set_figwidth(14)
    f.set_figheight(8)

    return