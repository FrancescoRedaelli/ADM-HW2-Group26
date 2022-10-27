import pandas as pd   # Data manipulation and analysis
import numpy as np    # Scientific Computing

import matplotlib.pyplot as plt   # Visualization
import matplotlib.lines as mlines   # Visualization

from tqdm import tqdm   # Progress bar

from collections import defaultdict   # Dictionary with default value
from datetime import datetime   # Date and Time handling

def initialize_posts_reader():
    return pd.read_csv(r'D:\Data\instagram_posts.csv', delimiter='\t', chunksize=100000, converters={"profile_id": str, "location_id": str})

###[RQ3]

def retrieve_posts_time(posts_reader):

    '''
    Retrieve the posts publishing time
    '''

    posts_time = []

    for chunk in posts_reader:
        hour_series = pd.to_datetime(chunk.cts).dt.hour
        posts_time.extend(hour_series.tolist())

    return posts_time

def plot_posts_time(posts_time):

    '''
    Plot the daily temporal distribution of posts published
    '''

    # Plot
    f = plt.figure()
    plt.xticks(range(25), ["{}:00".format(x).zfill(5) for x in range(25)], rotation = 45)
    plt.ylabel("# posts published (Millions)", fontsize=14, labelpad=20)
    plt.xlabel("Time of the day", fontsize=14, labelpad=20)
    plt.title("Daily temporal distribution of posts published", fontsize=18, pad=15)
    plt.hist(posts_time, bins=range(25), color='#00ff00', ec="k")
    plt.yticks(plt.gca().get_yticks(), [round(x/1e6,1) for x in plt.gca().get_yticks()])
    f.set_figwidth(14)
    f.set_figheight(8)

    return

def plot_posts_time_custom(posts_time, time_intervals, ftype=""):

    '''
    Plot the distribution of posts published per time interval
    '''

    # Input: time intervals list; each time interval is of the form [Start, End)

    intervals_out = {}

    for t in time_intervals:
        # Compute #posts per time interval
        intervals_out[t] = len([x for x in posts_time if t[0] <= x < t[1]])

    # Sort dict by key (time interval) for visualization purposes
    intervals_out = dict(sorted(intervals_out.items()))
    # Build time interval strings
    intervals_list = ["{}:00".format(x[0]).zfill(5) + " - " + "{}:00".format(x[1]).zfill(5) for x in intervals_out.keys()]

    # Plot
    f = plt.figure()
    plt.xticks(range(len(time_intervals)), intervals_list, rotation = 45)
    plt.ylabel("# posts published{}".format(" (Millions)" if ftype!="top10" else ""), fontsize=14, labelpad=20)
    plt.xlabel("Time interval", fontsize=14, labelpad=20)
    plt.title("Distribution of posts published per time interval{}".format(" of the top10 profiles by #posts" if ftype!="top10" else ""), fontsize=18, pad=15)
    plt.bar(intervals_list, intervals_out.values(), color='#ff6600', ec="k")
    if ftype!="top10": plt.yticks(plt.gca().get_yticks(), [round(x/1e6,1) for x in plt.gca().get_yticks()])
    f.set_figwidth(14)
    f.set_figheight(8)

    return

###[RQ4]

def build_profile_posts_map(posts_reader):

    '''
    Build the profile_id-posts dictionary
    '''

    mymap = dict()

    for chunk in tqdm(posts_reader):
        ids, post_id, cts, nlikes, ncomm = chunk.profile_id.tolist(), chunk.post_id.tolist(), chunk.cts.tolist(), chunk.numbr_likes.tolist(), chunk.number_comments.tolist()
        for i in range(len(ids)):
            if str(ids[i]) in mymap: mymap[str(ids[i])].append([post_id[i], cts[i], nlikes[i], ncomm[i], str(ids[i])])
            else: mymap[str(ids[i])] = [[post_id[i], cts[i], nlikes[i], ncomm[i], str(ids[i])]]

    return mymap

def posts_by_target_profile_ids(profile_posts_map, profile_id):

    '''
    Retrieve the posts belonging to the target profile_id
    '''

    if str(profile_id) not in profile_posts_map: return []
    return profile_posts_map[str(profile_id)]

def posts_by_top_profile_ids(profile_posts_map, profiles_df, n):

    '''
    Retrieve the posts belonging to the n top posted profiles for which data is available
    '''

    # Get unique profiles id
    profiles_available = set(profiles_df.profile_id)

    posts = []

    for id in profiles_available:
        # Retrieve posts by current profile
        curr_posts = posts_by_target_profile_ids(profile_posts_map, id)
        # If less than n profiles analyzed, store posts
        if len(posts)<n: posts.append(curr_posts)
        else:
            # Check if #posts is strictly more than the #posts of the bottom posted profile stored
            if len(posts[n-1])<len(curr_posts):
                # Remove the posts by bottom posted profile
                posts.pop()
                # Store the posts by current profile
                posts.append(curr_posts)
                # Sort stored posts by length in descending order
                posts.sort(key=len, reverse = True)

    return posts

def plot_avg_stats_top10(posts):

    '''
    Plot the avg number of likes and comments of the top10 profiles by #posts
    '''

    # Input: lists of posts of the top10 profiles by #posts

    avg_likes = []
    avg_comments = []

    profiles_ids = []

    # Compute avg #likes and #comments for each profile
    for profile in posts:
        profiles_ids.append((profile[0]).pop())
        values = list(map(lambda x: sum(x)/len(x), zip(*[[x[2],x[3]] for x in profile])))
        avg_likes.append(values[0])
        avg_comments.append(values[1])

    # Plot
    f = plt.figure()
    xticks = range(len(avg_likes))
    plt.xticks(xticks, profiles_ids, rotation = 45)
    plt.ylabel("Average value", fontsize=14, labelpad=20)
    plt.xlabel("Profile ID", fontsize=14, labelpad=20)
    plt.title("Average number of likes and comments of the top10 profiles by #posts", fontsize=18, pad=15)
    plt.bar([x-0.2 for x in xticks], avg_likes, width=0.4, color='#00ccff', ec="k")
    plt.bar([x+0.2 for x in xticks], avg_comments, width=0.4, color='#ff0000', ec="k")
    plt.legend(["Likes", "Comments"], fontsize=14, loc="upper left")
    f.set_figwidth(14)
    f.set_figheight(8)

    return

def plot_posts_top10(posts, time_intervals):

    '''
    Plot the number of posts sent by top 10 profiles by #posts in the given interval in [RQ3]
    '''

    posts_time = []

    for profile in posts:
        hours = [pd.to_datetime(x[1]).hour for x in profile]
        posts_time.extend(hours)

    plot_posts_time_custom(posts_time, time_intervals, ftype="top10")

    return

###BP(a)

def plot_two_categories(profiles_df, posts_reader):

    '''
    Plot the mean of time intervals between posts for top 10% profiles by #followers VS bottom 90%
    '''

    # Sort profiles by number of followers (Descending)
    profiles_df = profiles_df.sort_values(by="followers", ascending=False)

    # Compute the 10% number of profiles
    top10_value = int(len(profiles_df)/10)

    # Divide profiles by followers in two sets (for fast existence checking)
    top_profiles = set(profiles_df.head(top10_value)["profile_id"])
    bottom_profiles = set(profiles_df.tail(len(profiles_df)-top10_value)["profile_id"])

    # Retrieve posts times for the two categories
    top_dict = defaultdict(list)
    bottom_dict = defaultdict(list)

    times_list = []

    for chunk in posts_reader:
        ids, cts = chunk.profile_id.tolist(), chunk.cts.tolist()
        times_list.extend([(cts[i], ids[i]) for i in range(len(ids))])

    # Remove NaNs
    times_list = [x for x in times_list if str(x[0])!="nan"]

    # Sort by posts times
    date_format = "%Y-%m-%d %H:%M:%S.%f"
    times_list.sort(key=lambda x: datetime.strptime(str(x[0]), date_format))

    # Create dict of (sorted) posts times of each profile for each category
    for pair in times_list:
        if pair[1] in top_profiles: top_dict[pair[1]].append(pair[0])
        if pair[1] in bottom_profiles: bottom_dict[pair[1]].append(pair[0])

    # Compute time intervals
    top_intervals = []
    bottom_intervals = []

    for profile in top_dict.keys():
        top_intervals.extend(((pd.to_datetime(pd.Series(top_dict[profile]))).diff().dt.total_seconds().div(3600).tolist())[1:])

    for profile in bottom_dict.keys():
        bottom_intervals.extend(((pd.to_datetime(pd.Series(bottom_dict[profile]))).diff().dt.total_seconds().div(3600).tolist())[1:])

    # Plot
    f, ax = plt.subplots()
    boxplot = ax.boxplot([top_intervals, bottom_intervals], showmeans=True, notch=False, vert=True, patch_artist=True, labels=["Top 10%", "Bottom 90%"])
    for patch, color in zip(boxplot['boxes'], ['lightblue', 'pink']): patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_ylabel("Time Interval Length (Hours)", fontsize=14, labelpad=20)
    ax.set_xlabel("Category", fontsize=14, labelpad=20)
    ax.set_ylim(0, 700)
    green_triangle = mlines.Line2D([], [], color='green', marker='^', linestyle='None', markersize=14, label='Mean')
    ax.legend(handles=[green_triangle], fontsize=14)
    plt.title("Distribution of time intervals between posts (top 10% VS bottom 90% profiles)", fontsize=18, pad=15)
    f.set_figwidth(14)
    f.set_figheight(8)

    return