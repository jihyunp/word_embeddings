from RedditData import RedditData
from Word2VecUtils import plot_score_histogram, plot_three_scores_hist
#
# # CBOW model
# data_dir = './reddit_data_MH'
# word2vec_file = './word2vec_reddit_2008_2015_20/gensim_word2vec_2008_2015_orig.vectors'
# reddit_data = RedditData(data_dir, word2vec_file)
#
# n_test = 100
# within_subreddit, within_post, random_doc, comments_cbow = reddit_data.get_three_scores(n_test)
#
# plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
# plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
# plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')
#
# plot_three_scores_hist([within_subreddit, within_post, random_doc],
#                        ['within Subreddit', 'within Post', 'with Random Docs'],
#                        'wmd_three_hist_cbow.pdf')

# Skip gram
data_dir = './reddit_data_MH'
word2vec_file = './word2vec_reddit_sg_2008_2015_20/gensim_word2vec_sg_2008_2015_orig.vectors'
word2vec_file_bin = './word2vec_reddit_sg_2008_2015_20/skipgram_binary.vectors'
reddit_data = RedditData(data_dir, word2vec_file_bin, binary=True)

n_test = 100
within_subreddit, within_post, random_doc, comments_sg = reddit_data.get_three_scores(n_test)

plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')

plot_three_scores_hist([within_subreddit, within_post, random_doc],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       'wmd_three_hist_sg.pdf')


res1 = reddit_data.get_most_and_least_similar_comments_within_post(n_test=10, n_most=3, n_least=3,
                                                                   print_result=True)


test_sent = 'distributed representations of words and phrases and their compositionality'
res2 = reddit_data.get_most_and_least_similar_comments_custom(test_sent, n_pool=50,
                                                             n_most=3, n_least=3,
                                                             subreddit=None, verbose=True,
                                                             print_result=True)

comment_idx = 1179141 #1179141, 382519, 2040322
res3 = reddit_data.get_most_and_least_similar_comments_within_post_by_idx(comment_idx)

string1 = "they're already disadvantage missing classes. he's making anything difficult can't attend making easier can. helping students pass teacher's job."
string2 = string1 + "comment removed includes sort insulting discriminating word. bot r automoderator comments q11pu automoderator action performed automatically. please contact moderators subreddit message compose 2fr 2fgetmotivated questions concerns."
reddit_data.get_wmdistance_str(string1, string2)

