### Libraries & Setup

import pandas as pd   # Data manipulation and analysis
import numpy as np    # Scientific Computing

import matplotlib.pyplot as plt   # Visualization
import matplotlib.lines as mlines   # Visualization

import time   # Execution time measurement
from tqdm import tqdm   # Progress bar
import re   # Regular Expressions

import statistics as st # Stats
from statsmodels.regression.linear_model import OLS   # Regression
from statsmodels.distributions.empirical_distribution import ECDF   # Empirical CDF

from collections import defaultdict   # Dictionary with default value
from datetime import datetime   # Date and Time handling

import warnings
warnings.filterwarnings('ignore')

def initialize_posts_reader():
    return pd.read_csv(r'D:\Data\instagram_posts.csv', delimiter='\t', chunksize=100000, converters={"profile_id": str, "location_id": str})

# Reading csv in chunks
def read_csv(cols, path, num_chunks, chunksize=10**6, parse_dates=None, converters=None):
    '''
    Read csv in chunks and return a list of dataframes
    Input:
        cols: list of columns to read
        path: path to csv file
        num_chunks: number of chunks to read
        chunksize: number of rows to read in each chunk
        parse_dates: list of columns to parse as dates
    Output:
        df: dataframe
    '''

    if cols is None:
        cols = pd.read_csv(path, sep='\t', nrows=3).columns

    df = pd.DataFrame() # empty dataframe
    with pd.read_csv(path, sep='\t', usecols=cols, parse_dates=parse_dates, chunksize=chunksize, converters=converters) as reader:
        i = 1
        for chunk in reader:
            #print('Chunk', i)
            df = pd.concat([df, chunk], ignore_index=True)
            if i == num_chunks:
                break
            i += 1
    #print('Read', df.shape[0], 'rows and columns from', path+'!')
    return df

###[RQ1]

def EDA(df):
    '''
    Perform exploratory data analysis on the dataframe
    Input:
        df: dataframe
    '''
    print("The dimensions of the dataset: ", df.shape)
    print("The names of the columns: ", ', '.join(df.columns))
    missing = df.isnull().sum() / df.shape[0] * 100
    print("Columns with missing values and their percentage of missing data: ")
    print(missing[missing > 0].to_string(), end='\n\n')
    print("Possible categorical values in the data: ", df.nunique()[df.nunique() < 100].to_string(), end='\n\n')
    print("Number of duplicates in the data: ", df.duplicated().sum(), end='\n\n')
    # Descriptive statistics for only if followers, following, n_posts, numbr_likes, number_comments
    cols = df.columns[df.columns.isin(['followers', 'following', 'n_posts', 'number_likes', 'number_comments'])]
    if cols.size:
        print(df[cols].describe().to_string())

def EDA_for_posts():
    '''
    Perform exploratory data analysis on the posts dataframe
    '''
    path = r'D:\Data\instagram_posts.csv'
    cols = pd.read_csv(path, sep='\t', nrows=3).columns
    info = pd.DataFrame(index=cols, columns=['dtype', 'n_unique', 'missing_pct', 'min', 'max', 'mean', 'std'])
    for col in cols:
        df = read_csv([col], path, 43)
        info.loc[col, 'dtype'] = df.dtypes[0]

        if col == 'post_type':
            info.loc[col, 'n_unique'] = df.nunique()[0]
        info.loc[col, 'missing_pct'] = df.isnull().sum()[0] / df.shape[0] * 100

        if col in ['numbr_likes', 'number_comments']:
            info.loc[col, 'min'] = df.min()[0]
            info.loc[col, 'max'] = df.max()[0]
            info.loc[col, 'mean'] = df.mean()[0]
            info.loc[col, 'std'] = df.std()[0]

        nrows = df.shape[0]
        del df # remove the dataframe from memory

    print(f'The dataset has {nrows} rows and {cols.size} columns.')
    print(f'The names of the columns are: '+ ', '.join(cols))
    print("Columns with missing values and their percentage of missing data: ")
    print(info.loc[info['missing_pct'] > 0, 'missing_pct'].to_string(), end='\n\n')
    print("Possible categorical values in the data: ", info.loc[info['n_unique'] < 100, 'n_unique'].to_string(), end='\n\n')
    print("Descriptive statistics: ")
    print(info.loc[info.index.isin(['followers', 'following', 'n_posts', 'numbr_likes', 'number_comments']), ['min', 'max', 'mean', 'std']].to_string())

# Histogram for followıng, followers and n_posts, numbr_likes, number_comments, ignore nans, up to 95th percentile, stretch the plot
def plot_hists_eda():
    '''
    Plot histograms for following, followers, n_posts, numbr_likes, number_comments
    '''
    posts_df = read_csv(['profile_id', 'numbr_likes', 'number_comments'], r'D:\Data\instagram_posts.csv', 43)
    profiles_df = read_csv(['profile_id', 'followers', 'following', 'n_posts'], r'D:\Data\instagram_profiles.csv', 6)

    print('Standard scale histograms up till 95th percentile')
    fig, ax = plt.subplots(1, 5, figsize=(18, 4))
    ax[0].hist(profiles_df.following.dropna(), bins=100, range=(0, profiles_df.following.quantile(0.95)), density=True)
    ax[0].set_title('# Following')
    ax[1].hist(profiles_df.followers.dropna(), bins=100, range=(0, profiles_df.followers.quantile(0.95)), density=True)
    ax[1].set_title('# Followers')
    ax[2].hist(profiles_df.n_posts.dropna(), bins=100, range=(0, profiles_df.n_posts.quantile(0.95)), density=True)
    ax[2].set_title('# Posts')
    ax[3].hist(posts_df.numbr_likes.dropna(), bins=100, range=(0, posts_df.numbr_likes.quantile(0.95)), density=True)
    ax[3].set_title('# Likes')
    ax[4].hist(posts_df.number_comments.dropna(), bins=100, range=(0, posts_df.number_comments.quantile(0.95)), density=True)
    ax[4].set_title('# Comments')
    # show plot in center and stretch
    fig.tight_layout()
    plt.show()

    print('Log scale histograms up till 95th percentile')
    fig, axs = plt.subplots(1, 5, figsize=(18, 4))
    axs[0].hist(profiles_df.followers.dropna().apply(lambda x: np.log10(x+1)), bins=6, range=(0, 6), density=True)
    axs[0].set_title('Number of Followers per Profile')
    axs[0].set_xlabel('log10(followers)')
    axs[1].hist(profiles_df.following.dropna().apply(lambda x: np.log10(x+1)), bins=5, range=(0, 5), density=True)
    axs[1].set_title('Numer of Following per Profile')
    axs[1].set_xlabel('log10(following)')
    axs[2].hist(profiles_df.n_posts.dropna().apply(lambda x: np.log10(x+1)), bins=5, range=(0, 5), density=True)
    axs[2].set_title('Number of Posts per Profile')
    axs[2].set_xlabel('log10(n_posts)')
    axs[3].hist(posts_df.numbr_likes.dropna().apply(lambda x: np.log10(x+1)), bins=5, range=(0, 5), density=True)
    axs[3].set_title('Number of Likes per Post')
    axs[3].set_xlabel('log10(numbr_likes)')
    axs[4].hist(posts_df.number_comments.dropna().apply(lambda x: np.log10(x+1)), bins=5, range=(0, 5), density=True)
    axs[4].set_title('Number of Comments per Post')
    axs[4].set_xlabel('log10(number_comments)')
    fig.tight_layout()
    plt.show()

###[RQ2]

## Since the number of posts is object of analysis, we fill the NaN values
## Where not specified, number posts will be set at 0
def adjustProfDf(df):
    df.n_posts.fillna(0, inplace=True)
    return df

def computeCommentsLikesLocationAndType():
    most_liked_posts = []
    least_commented_posts = []
    most_commented_posts = []
    pic_video = Counter()
    location_counter = Counter()

    #At every iteration:
    # Fill NaN value for the set of fields we are interest in
    # Compute the chunk most liked, most commented and least commented, we append the result to a list
    # Count the photos only posts and the mixed type post
    # Count the posts where the location was registered
    for chunk in pd.read_csv(r'D:\Data\instagram_posts.zip', delimiter='\t', chunksize = 500000):
        chunk = adjustPostDf(chunk)
        most_liked_posts.append(chunk.sort_values(by='numbr_likes', ascending = False).head(10))
        least_commented_posts.append(chunk.sort_values(by = 'number_comments', ascending = True).head(10))
        most_commented_posts.append(chunk.sort_values(by = 'number_comments', ascending = False).head(10))
        pic_video['1'] += len(chunk.loc[chunk['post_type']==1])
        pic_video['3'] += len(chunk.loc[chunk['post_type']==3])
        location_yes = len(chunk.loc[chunk['location_id'] != ''])
        location_counter['Y'] += location_yes
        location_counter['N'] += (len(chunk) - location_yes)

    # We concate the results, so that we get just one DataFrame rather than a list of DataFrames
    most_liked_posts = pd.concat(most_liked_posts)
    least_commented_posts = pd.concat(least_commented_posts)
    most_commented_posts = pd.concat(most_commented_posts)
    return {'mostLikedPosts': most_liked_posts, 'mostCommentedPosts':most_commented_posts, 
            'leastCommentedPosts':least_commented_posts, 'typeCounter':pic_video, 
            'locationCounter':location_counter}

def getBusinessAccountPercentage(data_profiles):
    business_counter = Counter()
    # Count business accounts by filtering the DataFrame by 'is_business_account' (only True value will be taken).
    # We will do the same thing for non-business accounts. As for NaN values, we will derive them by substracting
    # the two previously computed values from the total profile amount

    n_profiles = len(data_profiles)
    business_counter['Y'] = len(data_profiles.loc[data_profiles['is_business_account'] == True])
    business_counter['N'] =  len(data_profiles.loc[data_profiles['is_business_account'] == False])
    business_counter['U'] = n_profiles - business_counter['Y'] - business_counter['N']

    #Compute percentage

    business_perc = business_counter['Y']*100/n_profiles
    non_business_perc = business_counter['N']*100/n_profiles
    NaN_business_perc = business_counter['U']*100/n_profiles
    
    #Round by 4 decimal points
    business_perc = round(business_perc,4)
    non_business_perc = round(non_business_perc,4)
    NaN_business_perc = round(NaN_business_perc,4)
    
    # Now I compute business accounts vs non business accounts without keeping count of NaN values
    total_without_NaN = n_profiles - business_counter['U']
    business_perc_withoutNaN = round(business_counter['Y']*100/total_without_NaN, 4)
    non_business_perc_withoutNaN = round(business_counter['N']*100/total_without_NaN, 4)

    return (business_perc, non_business_perc,NaN_business_perc, 
            business_perc_withoutNaN, non_business_perc_withoutNaN)

def plot_location_registered(location_counter):
    labels = 'Location Registered', 'No Location Registered'
    tot = location_counter['Y'] + location_counter['N']
    sizes = [round(location_counter['Y']*100/tot,4),round(location_counter['N']*100/tot,4)]
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

def plot_business_accounts_withNaN(business_perc, non_business_perc,NaN_business_perc):
    labels = 'Business Accounts', 'Non Business Accounts', 'NaN Business Accounts'
    sizes = [business_perc,non_business_perc,NaN_business_perc]
    explode = (0, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

def plot_business_accounts_withoutNaN(business_perc, non_business_perc):
    labels = 'Business Accounts', 'Non Business Accounts'
    sizes = [business_perc,non_business_perc]
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

def plot_n_posts(first_ten):

    '''
    Plot the profiles by number of posts
    '''
    # Input: profile DataFrame

    profiles_ids = []
    n_followers = []

    #Fill every list with info
    for x, profile in first_ten.iterrows():
        profiles_ids.append((profile['profile_id']))
        n_followers.append((profile['n_posts']))

    # Plot
    f = plt.figure()
    plt.xticks(range(len(first_ten)), profiles_ids, rotation = 45)
    plt.ylabel("Number of posts", fontsize=14, labelpad=20)
    plt.xlabel("Profile ID", fontsize=14, labelpad=20)
    plt.title("Number of posts for each profile", fontsize=18, pad=15)
    plt.bar([y for y in range(0,len(first_ten))], n_followers, width=0.8, color='#fa0000', ec="k")
    f.set_figwidth(14)
    f.set_figheight(8)

    return

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

###[RQ5]

def plot_n_followers(first_ten):
    
    '''
    Plot the profiles by number of FOLLOWERS
    '''

    # Input: profile DataFrame

    profiles_ids = []
    n_followers = []

    # Fill every list with info
    for x, profile in first_ten.iterrows():
        profiles_ids.append((profile['profile_id']))
        n_followers.append((profile['followers']))

    # Plot
    f = plt.figure()
    plt.xticks(range(10), profiles_ids)
    plt.ylabel("Number of followers", fontsize=14, labelpad=20)
    plt.xlabel("Profile ID", fontsize=14, labelpad=20)
    plt.title("Number of followers for each profile", fontsize=18, pad=15)
    plt.bar([y for y in range(0,10)], n_followers, width=0.8, color='#ff0000', ec="k")
    f.set_figwidth(14)
    f.set_figheight(8)
    return

def plot_most_visited_locations(most_location_id):    
    '''
    Plot the profiles by number of times a location has been visited
    '''
    labels = []
    sizes = []
    for x in most_location_id:
        loc_id, times_visited = x
        labels.append(loc_id)
        sizes.append(times_visited)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

def inf_posts(post_reader, inf_profile_id, inf_n_posts):
    '''
    "Get the most influential user's infos"
    '''

    # Input posts= dataFrame of posts
    inf_posts= []

    for chunk in post_reader:
        inf_posts.append(chunk.loc[chunk['profile_id'] == inf_profile_id])
        ## We do this in order to avoid useless iterations, if inf_post has inf_n_posts elements,
        ## we stop iterating since we have all the posts that we need
        if(len(inf_posts) == inf_n_posts):
            break        

    return pd.concat(inf_posts)

def countLikeAndCommentForPostType(inf_posts):
    ## Counter init for counting post_type
    post_type_counter = Counter({'1':0, '2':0, '3':0})
    ## Counter init for likes and comments
    like_by_type =  Counter({'1':0, '2':0, '3':0})
    comment_by_type =  Counter({'1':0, '2':0, '3':0})
    for post in inf_posts.itertuples():
        post_type_counter[str(post.post_type)]+=1
        like_by_type[str(post.post_type)]+=post.numbr_likes
        comment_by_type[str(post.post_type)]+=post.number_comments
    return {'PostTypeCounter':post_type_counter, 'LikeByType':like_by_type, 'CommentsByType':comment_by_type}

def calculatePostTypePercentage(post_type_counter, len_inf_posts):
    photo_percentage = round(post_type_counter['1'] * 100/len_inf_posts,4)
    video_percentage = round(post_type_counter['2'] * 100/len_inf_posts,4)
    multy_percentage = round(post_type_counter['3'] * 100/len_inf_posts,4)
    return (photo_percentage, video_percentage, multy_percentage)

def plot_post_by_type(photo_percentage, video_percentage, multy_percentage):    
    '''
    Plot post types
    '''
    labels = ['Photos', 'Videos', 'Mixed']
    sizes = [photo_percentage, video_percentage, multy_percentage]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=45)
    ax1.axis('equal') 

    plt.show()
    return

def plot_compared_to_followers(perc_likers, post_type, interaction_type):    
    '''
    Plot likes or comments compared to followers
    '''
    labels = ['Followers that ' + interaction_type + ' '+post_type, 
              'Followers that does not '+ interaction_type + ' ' +post_type]
    sizes = [perc_likers, 1-perc_likers]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.4f%%', startangle=45)
    ax1.axis('equal') 

    plt.show()
    return

def getMostVisitedLocations(most_location_id, locations):
    "Get the rows corrisponding to the most visited locations"
    most_visited_locations = []
    for index, chunk in locations.iterrows():
        if chunk['id'] in most_location_id:
            most_visited_locations.append(locations.iloc[[index]])
        ## We do this in order to avoid useless iterations, when we found the locations we need we stop the cycle
        if(len(most_visited_locations) == len(most_location_id) ):
            break
    return pd.concat(most_visited_locations)

###[RQ6]

def time_avg(df):

    date_format = "%Y-%m-%d %H:%M:%S.%f"
    dictio = df.groupby("profile_id")["cts"].apply(list).to_dict()
    sdictio = {k: v for k,v in sorted(dictio.items(), key = lambda v: len(v[1]), reverse = True)}
    values_1 = {i for i in sdictio if len(sdictio[i])<=1}
    for i in values_1:
        del sdictio[i]

    #each user gets its average time delta between posts

    pfreq = {}
    for i in sdictio.keys():
        temp = []
        for j in range(1,len(sdictio[i])):
            u = datetime.strptime(sdictio[i][j],date_format)
            v = datetime.strptime(sdictio[i][j-1],date_format)
            temp.append((u-v).total_seconds())
        pfreq[i] = st.mean(temp)

    #find the total average

    allfreq = []
    for i in pfreq.keys():
        allfreq.append(pfreq[i])
    res = st.mean(allfreq)
    resmin = st.mean(allfreq)/60
    res_days = int(resmin/1440)
    res_min = int(abs(res_days*1440-resmin))
    print("The average time window between posts is:",res_days,"days and",res_min,"minutes")

def highest_frequency_profiles(df):

    date_format = "%Y-%m-%d %H:%M:%S.%f"
    dictio = df.groupby("profile_id")["cts"].apply(list).to_dict()
    sdictio = {k: v for k,v in sorted(dictio.items(), key = lambda v: len(v[1]), reverse = True)}
    values_1 = {i for i in sdictio if len(sdictio[i])<=1}
    for i in values_1:
        del sdictio[i]

    pfreq = {}
    for i in sdictio.keys():
        temp = []
        for j in range(1,len(sdictio[i])):
            u = datetime.strptime(sdictio[i][j],date_format)
            v = datetime.strptime(sdictio[i][j-1],date_format)
            temp.append((u-v).total_seconds())
        pfreq[i] = st.mean(temp)

    #dictionary sorted by frequency

    spfreq = {k: v for k,v in sorted(pfreq.items(), key = lambda v: v[1], reverse = False)}
    profiles_2 = [i for i in spfreq.keys()]
    top3_2 = [profiles_2[0],profiles_2[1],profiles_2[2]]

    topfreq_2 = []
    for i in top3_2:
        topfreq_2.append(spfreq[i])

    return([(profiles_2[i],topfreq_2[i]) for i in range(3)])

def highest_freqs():
    profiles_df = read_csv(['profile_id', 'profile_name','followers', 'following', 'n_posts'], 'instagram_profiles.csv', 6)
    posts_df = read_csv(['profile_id', 'cts', 'numbr_likes', 'number_comments'], 'instagram_posts.csv', 43, parse_dates=['cts'])

    # List of matching profile ids
    mask_profile_ids = profiles_df['profile_id'].isin(posts_df['profile_id'].unique())
    profiles_df = profiles_df[mask_profile_ids]

    per_profile = pd.DataFrame()
    per_profile['n_posts_found'] = posts_df.groupby('profile_id').size()
    # Drop profiles with size smaller than 1
    per_profile = per_profile[per_profile['n_posts_found'] > 1]
    per_profile['time_span'] = (posts_df.groupby('profile_id')['cts'].max() - posts_df.groupby('profile_id')['cts'].min()).dt.total_seconds()
    # Keep profiles with time span larger than 0
    per_profile = per_profile[per_profile['time_span'] > 0]
    # post per day
    per_profile['freq'] = per_profile['n_posts_found'] / per_profile['time_span'] * 86400
    # Avg number of likes
    per_profile['avg_likes'] = posts_df.groupby('profile_id')['numbr_likes'].mean()
    # Avg number of comments
    per_profile['avg_comments'] = posts_df.groupby('profile_id')['number_comments'].mean()
    # Merge the two dataframes
    profiles_df = profiles_df.merge(per_profile, how='left', on='profile_id').sort_values('freq', ascending=False)
    profiles_df.drop('time_span', axis=1, inplace=True)

    print('Top 3 profiles that post the most frequently (with no NaNs):')
    print(profiles_df.dropna()[:3].to_string(), end='\n\n')

    print('Top 3 profiles that post the most frequently that had more than 10 posts found in the dataset (with no NaNs):')
    print(profiles_df[profiles_df['n_posts_found'] > 10].dropna()[:3].to_string())

    # Bar plot of the top 3 profiles that post the most frequently with and without more than 10 posts
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    profiles_df.dropna()[:3].plot.bar(x='profile_name', y='freq', ax=ax[0])
    profiles_df[profiles_df['n_posts_found'] > 10].dropna()[:3].plot.bar(x='profile_name', y='freq', ax=ax[1])
    ax[0].set_title('Top 3 profiles that post the most frequently')
    ax[1].set_title('Top 3 profiles that post the most frequently (more than 10 posts found)')
    ax[0].set_ylabel('Posts per day')
    ax[1].set_ylabel('Posts per day')
    # Set x axis angle to 60 degrees
    ax[0].tick_params(axis='x', labelrotation=60)
    ax[1].tick_params(axis='x', labelrotation=60)
    plt.show()

def plot_time_intervals_stats(df):

    #build 3 dictionaries for likes, comments, # posts

    hour_likes_dict = df.set_index(pd.to_datetime(df.cts).dt.hour)["numbr_likes"].to_dict()
    hour_comments_dict = df.set_index(pd.to_datetime(df.cts).dt.hour)["number_comments"].to_dict()

    posts_time = []
    hour_series = pd.to_datetime(df.cts).dt.hour
    posts_time.extend(hour_series.tolist())
    hour_posts_dict = {}
    for t in range(24):
        hour_posts_dict[t] = len([x for x in posts_time if t <= x < t+1])

    #construct arrays with average likes and comments in each time interval

    time_intervals = [(6,11), (11,14), (14,17), (17,20), (20,24), (0,3), (3,6)]

    results_likes = []
    results_comm = []
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(6,11):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(11,14):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(14,17):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(17,20):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(20,24):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(0,3):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)
    sum_p = 0
    sum_l = 0
    sum_c = 0
    for i in range(3,6):
        sum_p = sum_p + hour_posts_dict[i]
        sum_l = sum_l + hour_likes_dict[i]
        sum_c = sum_c + hour_comments_dict[i]
    results_likes.append(sum_l/sum_p)
    results_comm.append(sum_c/sum_p)

    #operations for visualization

    intervals_out = {}
    for t in time_intervals:
        intervals_out[t] = len([x for x in posts_time if t[0] <= x < t[1]])
    intervals_out = dict(sorted(intervals_out.items()))
    intervals_list = ["{}:00".format(x[0]).zfill(5) + " - " + "{}:00".format(x[1]).zfill(5) for x in intervals_out.keys()]

    #plot likes

    f = plt.figure()
    plt.bar(intervals_list, results_likes, color='#ff6600', ec="k")
    plt.ylabel("Average # of likes", fontsize=14, labelpad=20)
    plt.xlabel("Time interval", fontsize=14, labelpad=20)
    f.set_figwidth(14)
    f.set_figheight(8)

    #plot comments

    f = plt.figure()
    plt.bar(intervals_list, results_comm, color='#ff6600', ec="k")
    plt.ylabel("Average # of posts published{}".format(" (e-5)"), fontsize=14, labelpad=20)
    plt.xlabel("Time interval", fontsize=14, labelpad=20)
    f.set_figwidth(14)
    f.set_figheight(8)

###[RQ7]

def likes_to_follower():
    '''
    Plot the number of likes per follower
    '''
    posts_df = read_csv(['profile_id', 'numbr_likes'], r'D:\Data\instagram_posts.csv', 43)
    profiles_df = read_csv(['profile_id', 'followers'], r'D:\Data\instagram_profiles.csv', 6)

    # drop nan values
    posts_df = posts_df.dropna()
    profiles_df = profiles_df.dropna()

    posts_df.shape, profiles_df.shape

    # Add followers column to posts_df
    posts_df = posts_df.merge(profiles_df, on='profile_id', how='left')
    # Drop nan values
    posts_df.dropna(inplace=True)

    # likes to followers ratio
    x = posts_df['numbr_likes'] / posts_df['followers']
    x.dropna(inplace=True)

    # Empirical CDF of likes to followers ratio
    x = np.sort(x)
    y = np.arange(1, len(x)+1) / len(x)

    # Two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ECDF
    ax1.plot(x, y, marker='.', linestyle='none', alpha=0.1)
    ax1.title.set_text('ECDF of likes to followers ratio')
    ax1.set_xlabel('Likes to followers ratio')
    ax1.set_ylabel('ECDF')
    # Add vertical line at 0.2
    ax1.axvline(x=0.2, color='red', label='Likes to followers ratio = 0.2', linestyle='--')
    # Add horizontal line at F(0.2)
    ax1.axhline(y=y[sum(x<0.2)], color='green', label='F(0.2) = 0.2', linestyle='--')
    ax1.legend()

    # Plot ECDF for x < 0.4
    ax2.plot(x[x < 0.8], y[x < 0.8], marker='.', linestyle='none', alpha=0.1)
    ax2.title.set_text('ECDF of likes to followers ratio for x < 0.8')
    ax2.set_xlabel('Likes to followers ratio')
    ax2.set_ylabel('ECDF')
    # Add vertical line at 0.2
    ax2.axvline(x=0.2, color='red', label='Likes to followers ratio = 0.2', linestyle='--')
    # Add horizontal line at F(0.2)
    ax2.axhline(y=y[sum(x<0.2)], color='green', label=f'F(0.2) = {y[sum(x<0.2)]:.2f}', linestyle='--')
    ax2.legend()

    fig.tight_layout()
    plt.show()

    print(f'{100 * np.sum(x > 0.2) / len(x):.2f}% of the posts have a likes to followers ratio greater than 0.2.')

def return_to_locs():
    '''
    Calculate the rate of returning to locations for first post of each profile. Then plot the ECDF, and
    histogram by the number of posts profile.
    '''
    posts_df = read_csv(['profile_id', 'cts', 'location_id'], r'D:\Data\instagram_posts.csv', 43, parse_dates=['cts'])

    # Sort posts by profile_id and cts
    posts_df.sort_values(by=['profile_id', 'cts'], inplace=True)
    # Group by profile_id
    grouped = posts_df.groupby('profile_id')

    per_profile = pd.DataFrame()
    per_profile['first_post'] = grouped.first().cts
    per_profile['last_post'] = grouped.last().cts
    per_profile['num_posts'] = grouped.size()

    # Calculate the time between first and last post
    per_profile['time_between'] = per_profile.last_post - per_profile.first_post
    # Is it bigger than 24 hours?
    per_profile['time_between_gt_24h'] = per_profile.time_between > np.timedelta64(1, 'D')
    # Drop first and last post
    per_profile.drop(['time_between','first_post', 'last_post'], axis=1, inplace=True)
    # First location
    per_profile['first_location'] = grouped.first().location_id

    per_profile['first_location_visited_later24h']=False

    # Is the first location visited later than 24 hours after the first post?
    for prfl, group in grouped:
        if per_profile.loc[prfl, 'time_between_gt_24h']:
            # Locations after 24 hours
            locs_after_24h = group[group.cts > group.cts.iloc[0] + np.timedelta64(1, 'D')].location_id
            per_profile.loc[prfl, 'first_location_visited_later24h'] = per_profile.loc[prfl, 'first_location'] in locs_after_24h.values

    print(f'Percentage of profiles that visited the first location later than 24 hours: {per_profile[per_profile.time_between_gt_24h].first_location_visited_later24h.mean()*100:.2f}%')

    # Group by number of posts in intervals of 5
    # Two plots: one for 24 hours, one histogram of the number of posts
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15, 5))
    grouped_by_num_posts = per_profile.groupby(pd.cut(per_profile.num_posts, np.arange(0, 25)))

    # Plot the percentage of profiles that visited the first location later than 24 hours
    ax1.plot(grouped_by_num_posts.first_location_visited_later24h.mean().values*100, marker='o')
    ax1.set_title('Percentage of profiles that visited the first location later than 24 hours')
    ax1.set_xlabel('Number of posts')
    ax1.set_ylabel('Percentage (%)')
    #ax1.set_xticklabels(np.arange(0, 85, 5))

    # Plot eCDF of the number of posts
    ax2.plot(per_profile.num_posts.value_counts().sort_index().cumsum()/per_profile.shape[0]*100, marker='.', alpha=0.5)
    ax2.set_title('eCDF of the number of posts')
    ax2.set_xlabel('Number of posts')
    ax2.set_ylabel('Percentage (%)')

    fig.tight_layout()
    plt.show()

###[RQ8]

def scatter_likes_comments():
    '''
    Scatter plot of likes and comments
    '''
    posts_df = read_csv(['numbr_likes', 'number_comments'], r'D:\Data\instagram_posts.csv', 43)

    # Create df without outliers, i.e., in range of 3 standard deviations, numbr_likes and number_comments
    mask = (posts_df['numbr_likes'] < posts_df['numbr_likes'].mean() + 3 * posts_df['numbr_likes'].std()) & (posts_df['numbr_likes'] > posts_df['numbr_likes'].mean() - 3 * posts_df['numbr_likes'].std()) & (posts_df['number_comments'] < posts_df['number_comments'].mean() + 3 * posts_df['number_comments'].std()) & (posts_df['number_comments'] > posts_df['number_comments'].mean() - 3 * posts_df['number_comments'].std())
    posts_df_masked = posts_df[mask]
    # Scatter plot of number of likes (x) vs number of comments (y), unmasked and masked subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.scatter(posts_df['numbr_likes'], posts_df['number_comments'], marker='.', alpha=0.1)
    ax1.set_xlabel('Number of Likes')
    ax1.set_ylabel('Number of Comments')
    ax1.set_title(f'Entrire Dataset (n={posts_df.shape[0]})')
    ax2.scatter(posts_df_masked['numbr_likes'], posts_df_masked['number_comments'], marker='.', alpha=0.1)
    ax2.set_xlabel('Number of Likes')
    ax2.set_ylabel('Number of Comments')
    ax2.set_title(f'Outliers Removed (n={posts_df_masked.shape[0]})')
    fig.tight_layout()
    plt.show()



def testing_likes_comments():
    '''
    Testing whether a significant relationship exists between likes and comments
    '''
    posts_df = read_csv(['numbr_likes', 'number_comments'], r'D:\Data\instagram_posts.csv', 43)
    posts_df.dropna(inplace=True)

    # Create df without outliers, i.e., in range of 3 standard deviations, numbr_likes and number_comments
    mask = (posts_df['numbr_likes'] < posts_df['numbr_likes'].mean() + 3 * posts_df['numbr_likes'].std()) & (posts_df['numbr_likes'] > posts_df['numbr_likes'].mean() - 3 * posts_df['numbr_likes'].std()) & (posts_df['number_comments'] < posts_df['number_comments'].mean() + 3 * posts_df['number_comments'].std()) & (posts_df['number_comments'] > posts_df['number_comments'].mean() - 3 * posts_df['number_comments'].std())
    posts_df_masked = posts_df[mask]

    # Add constant to the dataframe
    posts_df['const'] = 1
    posts_df_masked['const'] = 1

    pd.options.mode.chained_assignment = None # To avoid warnings

    model1 = OLS(posts_df['number_comments'], posts_df[['const', 'numbr_likes']]).fit()
    posts_df_masked['numbr_likes_log'] = np.log(posts_df_masked['numbr_likes'] + 1)
    posts_df_masked['number_comments_log'] = np.log(posts_df_masked['number_comments'] + 1)
    model2 = OLS(posts_df_masked['number_comments_log'], posts_df_masked[['const', 'numbr_likes_log']]).fit()

    # Plot the scatter plot of number of likes (x) vs number of comments (y),for masked, and the regression lines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.scatter(posts_df_masked['numbr_likes'], posts_df_masked['number_comments'], marker='.', alpha=0.1)
    ax1.set_xlabel('Number of Likes')
    ax1.set_ylabel('Number of Comments')
    ax1.set_title(f'Level-Level Regression (BIC={model1.bic:.2f})')
    ax1.plot(posts_df_masked['numbr_likes'], model1.predict(posts_df_masked[['const', 'numbr_likes']]), color='red')
    ax2.scatter(posts_df_masked['numbr_likes_log'], posts_df_masked['number_comments_log'], marker='.', alpha=0.1)
    ax2.set_xlabel('Number of Likes (log)')
    ax2.set_ylabel('Number of Comments (log)')
    ax2.set_title(f'Log-Log Regression (BIC={model2.bic:.2f})')
    ax2.plot(posts_df_masked['numbr_likes_log'], model2.predict(posts_df_masked[['const', 'numbr_likes_log']]), color='red')
    # Add alpha and beta values with p-value in parenthesis to the plots
    ax1.text(0.05, 0.95, f'alpha={model1.params[0]:.2f} ({model1.pvalues[0]:.2f}), beta={model1.params[1]:.2f} ({model1.pvalues[1]:.2f})', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    ax2.text(0.05, 0.95, f'alpha={model2.params[0]:.2f} ({model2.pvalues[0]:.2f}), beta={model2.params[1]:.2f} ({model2.pvalues[1]:.2f})', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    fig.tight_layout()
    plt.show()



def followers_distrubution():
    '''
    Plotting the distribution of followers
    '''
    profiles_df = read_csv(['followers'], r'D:\Data\instagram_profiles.csv', 6)

    # Drop rows with na values on followers
    profiles_df.dropna(subset=['followers'], inplace=True)
    # sort the data by followers
    profiles_df.sort_values(by='followers', inplace=True, ignore_index=True)
    # Emprical CDF of followers
    eCDF = ECDF(profiles_df['followers'])
    # Plot the emprical CDF of followers, one with entire data and one that stops at 95th percentile
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(eCDF.x, eCDF.y)
    ax1.set_xlabel('Followers')
    ax1.set_ylabel('ECDF')
    ax1.set_title('Entire Dataset')
    ax2.plot(eCDF.x, eCDF.y)
    ax2.set_xlabel('Followers')
    ax2.set_ylabel('ECDF')
    ax2.set_title('Up Untill 95th Percentile')
    ax2.set_xlim(-500, eCDF.x[int(len(eCDF.x) * 0.95)])
    # Add the mean, median, q1, q3 as vertical lines
    ax2.axvline(np.mean(eCDF.x[1:]), color='green', linestyle='--', label='Mean')
    ax2.axvline(eCDF.x[int(len(eCDF.x) * 0.25)], color='yellow', linestyle='--', label='Q1')
    ax2.axvline(eCDF.x[int(len(eCDF.x) * 0.5)], color='orange', linestyle='--', label='Median')
    ax2.axvline(eCDF.x[int(len(eCDF.x) * 0.75)], color='red', linestyle='--', label='Q3')
    ax2.legend()
    plt.show()
    # Plot a single histogram of followers with approx line
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.hist(profiles_df['followers'][:int(profiles_df.shape[0] * 0.95)], bins=100)
    ax.set_xlabel('Number of Followers')
    ax.set_ylabel('Number of Profiles')
    ax.set_title('Distribution of Number of Followers for the First 95% of Profiles')
    # Add the mean, median, q1, q3 as vertical lines
    ax.axvline(np.mean(eCDF.x[1:]), color='green', linestyle='--', label='Mean')
    ax.axvline(eCDF.x[int(len(eCDF.x) * 0.25)], color='yellow', linestyle='--', label='Q1')
    ax.axvline(eCDF.x[int(len(eCDF.x) * 0.5)], color='orange', linestyle='--', label='Median')
    ax.axvline(eCDF.x[int(len(eCDF.x) * 0.75)], color='red', linestyle='--', label='Q3')
    ax.legend()
    fig.tight_layout()
    plt.show()

def boxplots():
    '''
    Example file for boxplots
    '''
    profiles_df = read_csv(['followers', 'n_posts'], r'D:\Data\instagram_profiles.csv', 6)

    # Example box and whiskers plot, partition the profiles by followers and plot the number of posts
    profiles_copy = profiles_df.copy()
    # Drop rows with na values on followers and n_posts
    profiles_copy.dropna(subset=['followers', 'n_posts'], inplace=True)
    # sort the data by followers
    profiles_copy.sort_values(by='followers', inplace=True, ignore_index=True)
    # Partition the data into 10 bins by percentiles
    profiles_copy['followers_bin'] = pd.qcut(profiles_copy['followers'], 10, labels=False)
    # Plot the box and whiskers plot no outliers
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.boxplot([profiles_copy[profiles_copy['followers_bin'] == i]['n_posts'] for i in range(10)], showfliers=False)
    ax.set_xlabel('Followers Bin')
    ax.set_ylabel('Number of Posts')
    ax.set_title('Number of Posts for Each Followers Bin')
    fig.tight_layout()
    plt.show()

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

###BP(b)

def same_day_same_week():
    posts_df = read_csv(['profile_id', 'location_id', 'cts'], r'D:\Data\instagram_posts.csv', 43, parse_dates=['cts'])

    # Save shape of posts_df
    shape = posts_df.shape
    # drop if cts is na
    posts_df = posts_df.dropna()
    #print(f'Number of rows dropped from posts dataset because of missing vals: {shape[0] - posts_df.shape[0]}', end='\n\n')

    posts_df['week'] = posts_df.cts.apply(lambda x: x.isocalendar()[0:2]) # (year, week)
    posts_df['date'] = posts_df.cts.apply(lambda x: x.strftime('%Y-%m-%d')) # (year, week, day)

    # Find posts that are published on the same date by the same profile in the same location and only show bigger than 1
    same_date = posts_df.groupby(['profile_id', 'location_id', 'date']).size().sort_values(ascending=False).loc[lambda x: x > 1]
    same_week = posts_df.groupby(['profile_id', 'location_id', 'week']).size().sort_values(ascending=False).loc[lambda x: x > 1]

    print(f'{len(same_date.index.get_level_values(0).unique())} of {len(posts_df.profile_id.unique())} profiles have posted more than once in the same location on the same day which makes {len(same_date.index.get_level_values(0).unique()) / len(posts_df.profile_id.unique()) * 100:.2f}% of all profiles.')
    print(f'{sum(same_date)} of {len(posts_df)} posts have been posted more than once in the same location on the same day which makes {sum(same_date) / len(posts_df) * 100:.2f}% of all posts.', end='\n\n')
    print(f'{len(same_week.index.get_level_values(0).unique())} of {len(posts_df.profile_id.unique())} profiles have posted more than once in the same location in the same week which makes {len(same_week.index.get_level_values(0).unique()) / len(posts_df.profile_id.unique()) * 100:.2f}% of all profiles.')
    print(f'{sum(same_week)} of {len(posts_df)} posts have been posted more than once in the same location in the same week which makes {sum(same_week) / len(posts_df) * 100:.2f}% of all posts.')

###BP(c)

def text_mining():
    # Read in the data
    descriptions = read_csv(['description'], r'D:\Data\instagram_posts.csv', 2).dropna().description

    descriptions = descriptions.apply(lambda x: x.replace(r'\n', ' ').lower())
    # Remove all russian and other chars
    descriptions = descriptions[descriptions.apply(lambda x: not bool(re.search('[\u0400-\u04FF,\u00C0-\u02AF]', x)))]
    descriptions = descriptions.apply(lambda x: re.sub(r'[^\x00-\x7F]+',' ', x)) # Remove all non ascii characters
    descriptions = descriptions.apply(lambda x: re.sub(r'\b\w{1,2}\b', '', x)) # Remove all words with 2 or less characters
    descriptions = descriptions.apply(lambda x: re.sub('[^a-z ]+', '', x.lower().strip())) # Remove all non a-z characters
    descriptions = descriptions.apply(lambda x: re.sub(' +', ' ', x)) # Sub all two or more spaces with one space
    descriptions = descriptions.apply(lambda x: re.sub(r'\b\w*([a-z])\1{2,}\w*\b', '', x)) # Remove all words with 3 or more repatative characters
    descriptions = descriptions[descriptions.apply(lambda x: len(x) > 3)] # Remove all words with 3 or less characters

    # merge a single regex
    search = ' la | el | los | las | de | del | en | y | que | dos | esse | e | al | una | il | di | mare | che | da | sempre | le | mi | amor | es | amo | pero | che | familia | por | te | feliz | tu | di | il | dan | ini | hari | una | che | yg | si | na | se | sa | ma | non | le | du |grazie|morgen'
    descriptions = descriptions[descriptions.apply(lambda x: not bool(re.search(search, x)))]
    # remove felices, para, siempre, vacaciones, ser, mis, somos, momentos, easter, pero, mis, en, route, mode, familia, pour, famille, casa, avec, weer, les,
    descriptions = descriptions[descriptions.apply(lambda x: not bool(re.search('felices| para |siempre|vacaciones|essere|questo| che | ser | mis | somos |momentos| pero | mis | en | route | mode | familia | pour | famille | casa | avec | weer | les', x)))]
    # remove med, dagen, og, dag, min, och, til, det, er, mai, meu, amor, com, amigo, deus, lugar, dia, bem, mundo, par, amici, friends, gli, serata, amicizia, miei, aperitivo, buona, cena, festa, met, een, van, het, today, deze, bij, people, mijn, voor
    descriptions = descriptions[descriptions.apply(lambda x: not bool(re.search(' med | dagen | og | dag | min | och | til | det | er | mai | meu | amor | com | amigo | deus | lugar | dia | bem | mundo | par | s | amici | friends | gli | serata | amicizia | miei | aperitivo | buona | cena | festa | met | een | van | het | today | deze | bij | people | mijn | voor', x)))]

    # Import the wordcloud library
    from wordcloud import WordCloud, STOPWORDS
    # Join the different processed titles together.
    long_string = ','.join(list(descriptions.values)[:10000])
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", stopwords=STOPWORDS, max_words=300, contour_width=3, contour_color='steelblue', width=1200, height=300)
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()
    # Show image
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    # Stretch the image
    plt.tight_layout()
    plt.show()

    # Text mining
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer

    # CountVectorizer, don't count words that appear in more than 0% of the documents, stop_words='english'
    #count_vect = CountVectorizer(min_df=100, max_df=0.01, stop_words='english', max_features=1000)
    count_vect = CountVectorizer(stop_words='english', min_df=100)
    X_train_counts = count_vect.fit_transform(descriptions)

    count_vect.get_feature_names_out()
    # Lemmatize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized = []
    for word in count_vect.get_feature_names_out():
        lemmatized.append(lemmatizer.lemmatize(word))
    # Remove duplicates
    lemmatized = list(set(lemmatized))

    len(lemmatized), len(count_vect.get_feature_names_out())
    lemmatized = [word for word in lemmatized if word not in ['abbiamo', 'abruzzo', 'io', 'essere', 'pero', 'che', 'questo', 'la', 'el', 'los', 'las', 'de', 'del', 'en', 'y', 'que', 'dos', 'esse', 'e', 'al', 'una', 'il', 'di', 'mare', 'che', 'da', 'sempre', 'le', 'mi', 'amor', 'es', 'amo', 'che', 'familia', 'por', 'te', 'feliz', 'tu', 'di', 'il', 'dan', 'ini', 'hari', 'una', 'che', 'yg', 'si', 'na', 'se', 'sa', 'ma', 'non', 'le', 'du', 'grazie', 'morgen', 'felices', 'para', 'siempre', 'vacaciones', 'ser', 'mis', 'somos', 'momentos', 'easter', 'pero', 'mis', 'en', 'route', 'mode', 'familia', 'pour', 'famille', 'casa', 'avec', 'weer', 'les', 'med', 'dagen', 'og', 'dag', 'min', 'och', 'til', 'det', 'er', 'mai', 'meu', 'amor', 'com', 'amigo', 'deus', 'lugar', 'dia', 'bem', 'mundo', 'par', 'amici', 'friends', 'gli', 'serata', 'amicizia', 'miei', 'aperitivo', 'buona', 'cena', 'festa']]

    # Remove words that are not in the lemmatized list from the count vectorizer
    count_vect = CountVectorizer(stop_words='english', vocabulary=lemmatized, min_df=100)
    X_train_counts = count_vect.fit_transform(descriptions)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    from sklearn.cluster import KMeans
    n_clusters = 25
    kmeans = KMeans(n_clusters=n_clusters, n_init=1, max_iter=35).fit(X_train_tfidf)

    # Print the top 10 words per cluster
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names_out()
    q = []
    for i in range(n_clusters):
        print("Cluster %d:" % i, f'n={np.where(kmeans.labels_ == i)[0].shape[0]}', end='  \t')
        # Print the number of words in the cluster
        q.append([terms[ind] for ind in order_centroids[i, :10]])
        print(', '.join([terms[ind] for ind in order_centroids[i, :10]]))

###[AQ2]

def plot_running_times(functions, input_sizes, alg=""):

    '''
    Plot the running time of the algorithms implemented for different input sizes
    '''

    times = []

    for func in functions:

        all_sizes_times = []

        for size in input_sizes:

            curr_size_times = []

            # Average result over 5 executions
            for i in range(5):

                # Start time
                start = time.process_time()

                # Execute function
                func(size)

                # End time
                end = time.process_time()

                # Compute execution time
                curr_size_times.append(end-start)

            all_sizes_times.append(sum(curr_size_times)/5)

        times.append(all_sizes_times)

    # Plot
    f = plt.figure()
    plt.xticks(input_sizes)
    plt.ylabel("Execution time (seconds)", fontsize=14, labelpad=20)
    plt.xlabel("Input size (N)", fontsize=14, labelpad=20)
    plt.title("Running time of algorithms for different input sizes", fontsize=18, pad=15)
    colors = ["Red", "Blue"]
    if alg == "DP": colors = ["Blue"]
    if alg == "AH": colors = ["Green"]
    for i in range(len(functions)): plt.plot(input_sizes, times[i], '-^', color=colors[i])
    plt.legend(["Recursive Algorithm", "DP Algorithm"], fontsize=14, loc="upper left")
    if alg == "DP": plt.legend(["DP Algorithm"], fontsize=14, loc="upper left")
    if alg == "AH": plt.legend(["Ad Hoc Algorithm"], fontsize=14, loc="upper left")
    plt.ylim(0)
    f.set_figwidth(14)
    f.set_figheight(8)

    return
