from load_data import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
import numpy as np
from word_movers_knn import *



if __name__ == "__main__":
    (vocab_dict, reverse_dict, W) = load_pretrained_word2vec()

    """
    d1 = "Obama speaks to the media in Illinois"
    d2 = "The President addresses the press in Chicago"

    vect = CountVectorizer(stop_words="english").fit([d1, d2])
    print("Features:", ", ".join(vect.get_feature_names()))


    # Copied from the pythonnotebook

    from scipy.spatial.distance import cosine

    v_1, v_2 = vect.transform([d1, d2])
    v_1 = v_1.toarray().ravel()
    v_2 = v_2.toarray().ravel()
    print(v_1, v_2)
    print("cosine(doc_1, doc_2) = {:.2f}".format(cosine(v_1, v_2)))




    from sklearn.metrics import euclidean_distances

    W_ = W[[vocab_dict[w] for w in vect.get_feature_names()]]
    D_ = euclidean_distances(W_)
    print("d(addresses, speaks) = {:.2f}".format(D_[0, 7]))
    print("d(addresses, chicago) = {:.2f}".format(D_[0, 1]))

    from pyemd import emd

    # pyemd needs double precision input
    v_1 = v_1.astype(np.double)
    v_2 = v_2.astype(np.double)
    v_1 /= v_1.sum()
    v_2 /= v_2.sum()
    D_ = D_.astype(np.double)
    D_ /= D_.max()  # just for comparison purposes
    print("d(doc_1, doc_2) = {:.2f}".format(emd(v_1, v_2, D_)))
    """


    # Subreddit doc classification

    docs, y, target_names = load_data_for_doc_class('./reddit_data_MH/tmp', num_cats=5)

    docs_train, docs_test, y_train, y_test = train_test_split(docs, y, train_size=300,
                                                              test_size=200, random_state=0)

    vect = CountVectorizer(stop_words="english").fit(docs_train + docs_test)
    common = [word for word in vect.get_feature_names() if word in vocab_dict]
    W_common = W[[vocab_dict[w] for w in common]]

    vect = CountVectorizer(vocabulary=common, dtype=np.double)
    X_train = vect.fit_transform(docs_train)
    X_test = vect.transform(docs_test)

    print('Training start')
    print(datetime.now())

    knn_cv = WordMoversKNNCV(cv=3, n_neighbors_try=range(1,15),
                             W_embed=W_common, verbose=5, n_jobs=3)
    knn_cv.fit(X_train, y_train)

    print("CV score: {:.2f}".format(knn_cv.cv_scores_.mean(axis=0).max()))

    print("Test score: {:.2f}".format(knn_cv.score(X_test, y_test)))



    #-------------------------------
    # Comparison with other models
    from sklearn.svm import LinearSVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.grid_search import GridSearchCV

    knn_grid = GridSearchCV(KNeighborsClassifier(metric='cosine', algorithm='brute'),
                            dict(n_neighbors=list(range(1, 20))),
                            cv=3)
    knn_grid.fit(X_train, y_train)
    print("CV score: {:.2f}".format(knn_grid.best_score_))
    print("Test score: {:.2f}".format(knn_grid.score(X_test, y_test)))

