from RedditData import RedditData, plot_score_histogram

data_dir = './reddit_data_MH/2015'
reddit_data = RedditData(data_dir)
n_test = 500

within_subreddit, within_post, random_doc = reddit_data.get_three_scores(n_test)

plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Subreddit', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')
