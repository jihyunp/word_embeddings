from RedditData import RedditData, plot_score_histogram

data_dir = './reddit_data_MH'
word2vec_file = './word2vec_reddit_2008_2015_20/gensim_word2vec_2008_2015_orig.vectors'
reddit_data = RedditData(data_dir, word2vec_file)

n_test = 100
within_subreddit, within_post, random_doc = reddit_data.get_three_scores(n_test)

plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')
