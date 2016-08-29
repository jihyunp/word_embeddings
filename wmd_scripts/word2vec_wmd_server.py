from RedditData import RedditData
from Word2VecUtils import plot_score_histogram, plot_three_scores_hist
from datetime import datetime

# CBOW model
# data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
# word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/gensim_word2vec_2008_2015.vectors'
# reddit_data = RedditData(data_dir, word2vec_file)
#
# n_test = 1000
# within_subreddit_cbow, within_post_cbow, random_doc_cbow, comments_cbow = reddit_data.get_three_scores(n_test)
#
# plot_score_histogram(within_subreddit_cbow, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
# plot_score_histogram(within_post_cbow, 'within Post', 'wmd_hist_within_post.pdf')
# plot_score_histogram(random_doc_cbow, 'with Random Docs', 'wmd_hist_random_doc.pdf')
#
# plot_three_scores_hist([within_subreddit_cbow, within_post_cbow, random_doc_cbow],
#                        ['within Subreddit', 'within Post', 'with Random Docs'],
#                        'wmd_three_hist_cbow.pdf')

# Skip gram
data_dir = '/extra/jihyunp0/research/word_embeddings/data/reddit_data_MH'
# word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/reddit_word2vec_sg/gensim_word2vec_sg_2008_2015.vectors'
word2vec_file = '/extra/jihyunp0/research/word_embeddings/result/reddit_word2vec_sg/skipgram_binary.vectors'
reddit_data= RedditData(data_dir, word2vec_file, binary=True)

n_test = 2000
within_subreddit, within_post, random_doc, comments_sg = reddit_data.get_three_scores(n_test)

plot_score_histogram(within_subreddit, 'within Subreddit', 'wmd_hist_within_subreddit.pdf')
plot_score_histogram(within_post, 'within Post', 'wmd_hist_within_post.pdf')
plot_score_histogram(random_doc, 'with Random Docs', 'wmd_hist_random_doc.pdf')

plot_three_scores_hist([within_subreddit, within_post, random_doc],
                       ['within Subreddit', 'within Post', 'with Random Docs'],
                       'wmd_three_hist_sg.pdf')


res1 = reddit_data.get_most_and_least_similar_comments_within_post(n_test=10, n_most=3, n_least=3,
                                                                   print_result=True)

# test_sent = 'distributed representations of words and phrases and their compositionality'
# res2 = reddit_data.get_most_and_least_similar_comments_custom(test_sent, n_pool=50,
#                                                              n_most=3, n_least=3,
#                                                              subreddit=None, verbose=True,
#                                                              print_result=True)
#
# comment_idx = 1179141 #1179141, 382519, 2040322
# res3 = reddit_data.get_most_and_least_similar_comments_within_post_by_idx(comment_idx)
#
#
# string1 = "they're already disadvantage missing classes. he's making anything difficult can't attend making easier can. helping students pass teacher's job."
# string2 = string1 + "comment removed includes sort insulting discriminating word. bot r automoderator comments q11pu automoderator action performed automatically. please contact moderators subreddit message compose 2fr 2fgetmotivated questions concerns."
# reddit_data.get_wmdistance_str(string1, string2)



















print('\nGetting the most and the least similar comments within the same thread')
t3 = datetime.now()
most_and_least = reddit_data_sg.get_most_and_least_similar_comments_within_post(100)
print('Took %.2f seconds' % (datetime.now() - t3).seconds)


for item in most_and_least[10:20]:
    print('----------------------------------------')
    rsent = item[0]
    most = item[1]
    least = item[2]
    print('Sent '+ str(rsent[0]) )
    print(rsent[2])
    print('\n')
    print('Most Similar Sent ' + str(most[0]))
    print('Score: ' + str(most[1]))
    print(most[2])
    print('\n')
    print('Least Similar Sent ' + str(least[0]))
    print('Score: ' + str(least[1]))
    print(least[2])


