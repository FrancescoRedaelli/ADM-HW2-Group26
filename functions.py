import pandas as pd   # Data manipulation and analysis
import numpy as np    # Scientific Computing

import matplotlib.pyplot as plt   # Visualization
from tqdm import tqdm   # Progress bar
from collections import Counter

def initialize_posts_reader():
    return pd.read_csv(r'C:\Users\PepeSa\Downloads\ADM\HMW2\instagram_posts.zip', delimiter='\t', chunksize=100000, converters={"profile_id": str, "location_id": str})

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

    for chunk in posts_reader:
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

    posts_time = []

    for profile in posts:
        hours = [pd.to_datetime(x[1]).hour for x in profile]
        posts_time.extend(hours)

    plot_posts_time_custom(posts_time, time_intervals, ftype="top10")

    return

## Used for RQ2
## Since the number of likes, comment and location is object of analysis, we fill the NaN values
def adjustPostDf(df):
    df.location_id.fillna('', inplace=True)
    df.numbr_likes.fillna(0, inplace=True)
    df.number_comments.fillna(0, inplace=True)
    return df

## Used for RQ2
## Since the number of posts is object of analysis, we fill the NaN values
## Where not specified, number posts will be set at 0
def adjustProfDf(df):
    df.n_posts.fillna(0, inplace=True)
    return df

#used For RQ2
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
    for chunk in pd.read_csv(r'C:\Users\PepeSa\Downloads\ADM\HMW2\instagram_posts.zip', delimiter='\t', chunksize = 500000):
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

#Used For RQ2
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

#used For RQ2
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

#used For RQ2
def plot_business_accounts_withNaN(business_perc, non_business_perc,NaN_business_perc):
    labels = 'Business Accounts', 'Non Business Accounts', 'NaN Business Accounts'
    sizes = [business_perc,non_business_perc,NaN_business_perc]
    explode = (0, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

#used For RQ2
def plot_business_accounts_withoutNaN(business_perc, non_business_perc):
    labels = 'Business Accounts', 'Non Business Accounts'
    sizes = [business_perc,non_business_perc]
    explode = (0.1, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal') 

    plt.show()
    return

##Used both for RQ5 & RQ2
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

##RQ5
def plot_n_followers(first_ten):
    
    '''
    Plot the profiles by number of FOLLOWERS
    '''

        # Input: profile DataFrame 

    profiles_ids = []
    n_followers = []

    #Fill every list with info
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

##RQ5
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

##RQ5
def inf_posts(post_reader, inf_profile_id, inf_n_posts):
    ##Input posts= dataFrame of posts
    "Get the most influential user's infos"

    inf_posts= []

    for chunk in post_reader:
        inf_posts.append(chunk.loc[chunk['profile_id'] == inf_profile_id])
        ## We do this in order to avoid useless iterations, if inf_post has inf_n_posts elements,
        ## we stop iterating since we have all the posts that we need
        if(len(inf_posts) == inf_n_posts):
            break        

    return pd.concat(inf_posts)

##RQ5
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

##RQ5
def calculatePostTypePercentage(post_type_counter, len_inf_posts):
    photo_percentage = round(post_type_counter['1'] * 100/len_inf_posts,4)
    video_percentage = round(post_type_counter['2'] * 100/len_inf_posts,4)
    multy_percentage = round(post_type_counter['3'] * 100/len_inf_posts,4)
    return (photo_percentage, video_percentage, multy_percentage)

##RQ5
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

##RQ5
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

#RQ5
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