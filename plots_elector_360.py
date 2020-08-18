import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import geopandas as gpd
import json


# plot metrics of df
def metrics_of_tweet(
                        df,
                        retweet_col='retweet',
                        conversation_id_col='conversation_id',
                        id_col='id',
                        retweet_id_col='retweet_id',
                        user_id_col='user_id',
                        user_rt_id_col='user_rt_id',
                    ):
    
    df = df.drop_duplicates(subset=[conversation_id_col, id_col, retweet_id_col])
    number_tweets = df.shape[0]
    number_users = len(set(i for i in list(df[user_id_col].unique()) + list(df[user_rt_id_col].unique())))
    number_tweets_orig = sum((df[conversation_id_col]==df[id_col]) & (df[retweet_col]==False))
    number_answers = sum((df[conversation_id_col]!=df[id_col]) & (df[retweet_col]==False))
    number_retweets = sum(df[retweet_col]==True)

    labels = ['Orig. Tweets', 'Answers', 'Retweets']
    values = [number_tweets_orig, number_answers, number_retweets]

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode = "number",
        value = number_tweets, 
        domain = {'x': [0, 0.5], 'y': [0.75, 1]},
        title = 'Number of Tweets'))

    fig.add_trace(go.Indicator(
        mode = "number",
        value = number_users,   
        domain = {'x': [0, 0.5], 'y': [0.12, 0.3]},
        title = 'Number of Users'))

    fig.add_trace(go.Indicator(
        mode = "gauge", 
        gauge = {
        'shape': "bullet",
        'axis': {'range': [None, number_tweets]}},
        value = number_users,
        domain = {'x': [0, 0.5], 'y': [0, 0.1]}))

    fig.add_trace(go.Pie(
        labels = labels, 
        values = values,
        textinfo = 'none', 
        hole = 0.7,
        domain = {'x': [0.65, 1], 'y': [0, 1]},
        title = 'Types of Tweets'))

    fig.update_layout(
        grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
        legend = {
                'x': 1, 'y': 0.5, 
                'orientation': 'v', 
                'yanchor': 'middle'},)
    

    return fig






# plot polygon with color
def geo_map(
            df,
            neighborhood_col,
            values_col,
            aggfunc_name,
            geojson_path,
            geojson_neighborhood_name,
            mapbox_accesstoken,
            fill_value=0,
            log_scale=False,
            threshold=0,
            colorscale=["#EF553B", "#FECB52", "#00CC96"],
            marker_opacity=0.5,
            lat=-31.4,
            lon=-64.2,
            mapbox_zoom=10,
            ):

    pivot_neighborhood = pd.pivot_table(
                                    data = df,
                                    index = neighborhood_col,
                                    values = values_col,
                                    aggfunc = aggfunc_name,
                                    fill_value = fill_value)\
                                .reset_index()\
                                .rename(columns = {
                                                values_col: 'values',
                                                neighborhood_col: 'neighborhood'})

    neighborhood_df = gpd.read_file(geojson_path)\
                            .dissolve(by = geojson_neighborhood_name, aggfunc = 'sum')\
                            .reset_index()
    neighborhood_df = pd.merge(neighborhood_df, pivot_neighborhood, 
                                left_on = geojson_neighborhood_name, 
                                right_on = neighborhood_col,
                                how = 'left')
    neighborhood_df = neighborhood_df[neighborhood_df['values']>threshold]

    if log_scale==True:
        neighborhood_df['values'] = np.log(neighborhood_df['values']).replace([np.inf, -np.inf], 0)

    neighborhood_json = json.loads(neighborhood_df.to_json())

    fig = go.Figure(go.Choroplethmapbox(
                    geojson = neighborhood_json, 
                    locations = neighborhood_df[geojson_neighborhood_name], 
                    z = neighborhood_df['values'],
                    featureidkey = str("properties." + geojson_neighborhood_name),
                    text = neighborhood_df["values"].astype(str),
                    colorscale = colorscale, 
                    marker_opacity = marker_opacity, 
                    marker_line_width = 0,
                    visible = True,
                    colorbar_title = str(values_col)))

    fig.update_layout(
                    margin = {"r":0,"t":0,"l":0,"b":0},
                    mapbox_accesstoken = mapbox_accesstoken,
                    mapbox_style = "carto-positron",
                    mapbox_center = {'lat': lat, 'lon': lon},
                    mapbox_zoom = mapbox_zoom)
    
    return fig


# Get the basic information about user 
def get_basics(tweets_final, df, user_name_col, user_id_col):
    tweets_final["screen_name"] = df[user_name_col]
    tweets_final["user_id"] = df[user_id_col]
    return tweets_final


# Get the user mentions 
def get_usermentions(tweets_final, df, user_mention_name_col, user_mention_id_col):
    # Inside the tag 'entities' will find 'user mentions' and will get 'screen name' and 'id'
    tweets_final["user_mentions_screen_name"] = df[user_mention_name_col]
    tweets_final["user_mentions_id"] = df[user_mention_id_col]
    return tweets_final


# Get retweets
def get_retweets(tweets_final, df, user_retweet_name_col, user_retweet_id_col):
    # Inside the tag 'retweeted_status' will find 'user' and will get 'screen name' and 'id'    
    tweets_final["retweeted_screen_name"] = df[user_retweet_name_col]
    tweets_final["retweeted_id"] = df[user_retweet_id_col]
    return tweets_final


# Get the information about replies
def get_in_reply(tweets_final, df, reply_to_col):
    df['reply_to'] = df[reply_to_col].str.replace("'",'"')
    # Just copy the 'in_reply' columns to the new dataframe
    tweets_final["in_reply_to_screen_name"] = df[reply_to_col].apply(lambda x: json.loads(x)[0]['username'] if x else np.nan)
    tweets_final["in_reply_to_user_id"]= df[reply_to_col].apply(lambda x: json.loads(x)[0]['user_id'] if x else np.nan)
    return tweets_final


# Get the interactions between the different users
def get_interactions(row, mention):
    # From every row of the original dataframe
    # First we obtain the 'user_id' and 'screen_name
    user = row["user_id"], row["screen_name"]
    # Be careful if there is no user id
    if user[0] is None:
        return (None, None), []
    
    # The interactions are going to be a set of tuples
    interactions = set()
    
    # Add all interactions 
    # First, we add the interactions corresponding to replies adding the id and screen_name
    interactions.add((row["in_reply_to_user_id"], row["in_reply_to_screen_name"]))
    # After that, we add the interactions with retweets
    interactions.add((row["retweeted_id"], row["retweeted_screen_name"]))
    # And later, the interactions with user mentions
    if mention == True:
        interactions.add((row["user_mentions_id"], row["user_mentions_screen_name"]))
    
    # Discard if user id is in interactions
    interactions.discard((row["user_id"], row["screen_name"]))
    # Discard all not existing values
    interactions.discard((None, None))
    # Return user and interactions
    return user, interactions


# get color sentiment for each node
def get_colors(
                df,
                graph_to_plot,
                ):

    negative = list(df[(df['sentiment_category']=='negative')]['screen_name'].unique())
    negative.extend(list(df[(df['sentiment_category']=='negative')]['retweeted_screen_name'].unique()))
    negative = [i for i in negative if i in graph_to_plot.nodes]

    positive = list(df[(df['sentiment_category']=='positive')]['screen_name'].unique())
    positive.extend(list(df[(df['sentiment_category']=='positive')]['retweeted_screen_name'].unique()))
    positive = [i for i in positive if i in graph_to_plot.nodes]

    neutral = [i for i in list(df['screen_name'].unique()) + list(df['retweeted_screen_name'].unique()) if i in graph_to_plot.nodes]
    neutral = [i for i in neutral if i not in negative + positive]

    central_nodes = negative + positive + neutral
    colors_central_nodes = ['red' for i in range(len(negative))]\
                            + ['green' for i in range(len(positive))]\
                            + ['yellow' for i in range(len(neutral))]

    return central_nodes, colors_central_nodes


# plot user networks
def network(
            df,
            id_col, # id of tweet
            user_name_col,
            user_id_col,
            user_retweet_name_col,
            user_retweet_id_col,
            reply_to_col,
            sentiment_category_col, # negative, neutral, positive
            network_type='all', # 'all' or 'largest_subgraph' 
            figsize=None, # (float, float) size of plot
            user_mention_name_col=None, # only one user mention
            user_mention_id_col=None, # only one user mention, requires user_mention_name_col
            ):
    
    columns = ["id", 
                "screen_name", "user_id",
                "retweeted_screen_name", "retweeted_id",
                "reply_to",
                "sentiment_category",
                "in_reply_to_screen_name", "in_reply_to_user_id"]

    # add mention network
    if user_mention_name_col != None or user_mention_id_col != None:
        mention = True
        columns.extend(["user_mentions_screen_name", "user_mentions_id"])
    else:
        mention = False

    tweets_final = pd.DataFrame(columns = columns)
    tweets_final[['id', 'sentiment_category']] = df[[id_col, sentiment_category_col]]
    tweets_final = get_basics(tweets_final, df, user_name_col, user_id_col)
    tweets_final = get_retweets(tweets_final, df, user_retweet_name_col, user_retweet_id_col)
    tweets_final = get_in_reply(tweets_final, df, reply_to_col)
    if mention == True:
        tweets_final = get_usermentions(tweets_final, df, user_mention_name_col, user_mention_id_col)
    tweets_final = tweets_final.where((pd.notnull(tweets_final)), None)

    graph = nx.Graph()

    for index, tweet in tweets_final.iterrows():
        user, interactions = get_interactions(tweet, mention)
        user_id, user_name = user
        tweet_id = tweet["id"]

        for interaction in interactions:
            int_id, int_name = interaction
            graph.add_edge(user_name, int_name, tweet_id=tweet_id)
            graph.nodes[user_name]["name"] = user_name
            graph.nodes[int_name]["name"] = int_name

    degrees = [val for (node, val) in graph.degree()]
    largest_subgraph = max(nx.connected_component_subgraphs(graph), key=len)
    graph_centrality = nx.degree_centrality(largest_subgraph)
    max_de = max(graph_centrality.items(), key=itemgetter(1))
    graph_closeness = nx.closeness_centrality(largest_subgraph)
    max_clo = max(graph_closeness.items(), key=itemgetter(1))
    graph_betweenness = nx.betweenness_centrality(largest_subgraph, normalized=True, endpoints=False)
    max_bet = max(graph_betweenness.items(), key=itemgetter(1))
    node_and_degree = largest_subgraph.degree()

    plt.figure(figsize=figsize)
    plt.axis("off")

    if network_type == 'all':

        central_nodes, colors_central_nodes = get_colors(tweets_final, graph)
        pos = nx.spring_layout(graph, k=0.5)
        nx.draw(graph, pos=pos, cmap=plt.cm.PiYG, edge_color="black", linewidths=0.3, alpha=0.5, node_size=0)
        nx.draw_networkx_nodes(graph, pos=pos, node_size=100, alpha=0.5, 
                                node_color=colors_central_nodes, nodelist=central_nodes)

        return plt

    elif network_type == 'largest_subgraph':

        central_nodes, colors_central_nodes = get_colors(tweets_final, largest_subgraph)
        pos = nx.spring_layout(largest_subgraph, k=0.05)
        nx.draw_networkx(largest_subgraph, pos=pos, cmap=plt.cm.PiYG, edge_color="black", linewidths=0.3, 
                            node_size=100, alpha=0.5, with_labels=True)
        nx.draw_networkx_nodes(largest_subgraph, pos=pos, node_size=100, alpha=0.5,
                            nodelist=central_nodes, node_color=colors_central_nodes)
    
        return plt

    else:
        return None


# clean text from list of words
def clean_list(x, stopwords):
    return list(set([clean_text(unidecode(token)) for token in x if str(token)!='nan' and clean_text(unidecode(token)) not in stopwords]))


# clean word
def clean_text(x):
    string = str(x)
    string = re.sub(r'[\W_]+', '', x) # remove anything that is not a letter or number
    string = string.strip() # remove spaces at the beginning and at the end of the string
    return  string.lower()


# plot bar most important topics with color
def ntrend_topics_bar(
                        df,
                        topics_col, # column with list of string
                        stopwords=[], # list of stopwords
                        ntop=20,
                        color_col=None, # column numeric values
                        aggfunc_name='mean', # function to color column
                        fill_value=0,
                        title=None,
                        color_continuous_scale=["#EF553B", "#FECB52", "#00CC96"],
                        range_color=[-1, 1],
                        ):

#     df['word_frec'] = df[topics_col].apply(lambda x: clean_list(literal_eval(x), stopwords))
    df['word_frec'] = df[topics_col]
    list_words = df['word_frec'].apply(pd.Series).stack().tolist()
    top = pd.Series(list_words).value_counts()[:ntop]

    if color_col is not None:
        words_top_color = df[[color_col, 'word_frec']]\
                                .set_index(color_col)\
                                .squeeze()\
                                .apply(pd.Series)\
                                .stack()\
                                .reset_index(1, drop=True)\
                                .reset_index()\
                                .rename(columns={0:'word_frec'})
        words_top_color = words_top_color.groupby(['word_frec'])[color_col].agg(aggfunc_name).reset_index()
        words_top_color = words_top_color[words_top_color['word_frec'].isin(top.index)]
        words_top_color = pd.merge(top.reset_index().rename(columns={'index':'word_frec', 0:'total'}),
                                    words_top_color, on='word_frec').fillna(fill_value)
        fig = px.bar(
                    data_frame = words_top_color, 
                    x = "total", 
                    y = "word_frec", 
                    color = color_col, 
                    orientation = 'h',
                    title = title,
                    color_continuous_scale = color_continuous_scale,
                    range_color = range_color,
                    )
        fig.update_layout(yaxis = {'categoryorder':'total ascending'})

        return fig

    else:
        fig = px.bar(
                    x = top.values, 
                    y = top.index,
                    orientation = 'h',
                    title = title,
                    )
        fig.update_layout(yaxis = {'categoryorder':'total ascending'})

        return fig