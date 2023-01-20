import pandas as pd
import scipy.sparse
from sklearn.feature_selection import SelectKBest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy import sparse

# Functions used for Logistic-Regression model development
# This class is based on:
#  - https://github.com/tasosnent/BeyondMeSH

MaxWeights = 'MaxWeights'

def getTFIDF(sample_counts, corpus_counts=None):
    '''
    Calculate the TFIDF value for each token in each document (in :param sample_counts:) based on token counts.
        The optinal :param corpus_counts: is only used for normalization (i.e. calculation of IDF) for TFIDF calculation.
        When called with one parameter, both TF and IF are calculated in the same set of documents.
        ATTENTION: Both csr_matrices should use the same vocabulary (i.e. the same ids for the features/tokens). In particular, it makes sense that both use a vocabulary based on :param sample_counts: articles.
    :param sample_counts:  Counts of the articles (e.g. test data) to calculate the TF. It is a scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
    :param corpus_counts:  (Optinal) Counts of the corpus (e.g. training data) to calculate the IDF. It is a scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
    :return:        A scipy.sparse.csr_matrix (size: num of documents in :param sample_counts:, num of tokens in :param corpus_counts:) with TFIDF value for each token-document combination, for all documents in :param sample_counts:
    '''

    # If no separate corpus_counts is provided, use sample_counts as the coprus for IDF calculation
    if corpus_counts is None:
        corpus_counts = sample_counts

    tfidf_transformer = TfidfTransformer()
    # Fit to corpus data
    tfidf_transformer.fit(corpus_counts)
    # Transform samples to a tf-idf matrix.
    train_tfidf = tfidf_transformer.transform(sample_counts)
    # Print sample No (articles) and feature No (tokens)
    # print(train_tfidf.shape)
    return train_tfidf

def tokenizeArticleText(X, vocabulary=None):
    '''
    Tokenize text data using a predefined vocabulary (:param vocabulary:) and produce a sparse representation of each token count in each document.
    If optional :param vocabulary: is not present then fit/create a vocabulary with all tokens present in the dataset
    Represent token counts as a scipy.sparse.csr_matrix with dimensions (number of documents, number of tokens)
    Based on this example : http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    :param X: An Array-like structure with text data to be tokenized. (i.e. one document per index)
    :param vocabulary: (Optional) The vocabulary used for tokenization as a Dictionary token/term to token index/id (As expected/produced by Tokenizer)
    :return: counts, feature_names
        1)  counts :  A scipy.sparse.csr_matrix (size: num of documents, num of token ids) with token count for each token-document combination
        2)  feature_names : A list (size: num of tokens) with each "token string" in the corresponding index. (So that the index of a token in feature_names corresponds to the id used in counts for this token)
    '''

    if not vocabulary is None:
        count_vect = CountVectorizer(vocabulary=vocabulary)
    else:
        count_vect = CountVectorizer()

    # Fit and return term-document matrix (transform).
    counts = count_vect.fit_transform(X)
    # Print sample No (articles) and feature No (tokens)
    # print(counts.shape)
    # print (sample no: article, feature no: word id)  number of occurences
    # print(counts)

    # get inverted vocabulary
    inv_vocabulary = {v: k for k, v in count_vect.vocabulary_.items()}
    # print(inv_vocabulary)

    # Create a list with the names of the features
    feature_names = []
    for i in range(len(count_vect.vocabulary_)):
        feature_names.insert(i, inv_vocabulary[i])
    return counts, feature_names

def getFeatureWeights(X, Y, tokens, labelsNames, scoreFunction):
    '''
    Calculate weights for the features of a dataset performing univariate feature selection based on :param scoreFunction:
        Weights are calculated per label and the Maximum weight (across labels) is selected to create an additional column
    :param X: ndarray (documents, tokens) with tfidf for each document-token combination
    :param Y: ndarray (documents, labels) with 1/0 for each document-label combination
    :param tokens: A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
    :param labelsNames: The names of the labels as a list of strings in the specified order used in :param Y: columns
    :param scoreFunction: The function to be used for scoring during feature selection [For classification: chi2, f_classif, mutual_info_classif]
    :return: The weights for each feature as a pandas.DataFrame with size (features, labels+2) with:
            - One column for each label (e.g. C0013264, C0917713, C3542021)
            - One column with the tokens ('token') and
            - One column with the maximum weight for each token across all labels
    '''
    # Select k best features
    # Create feature selector
    selector = SelectKBest(score_func=scoreFunction, k='all')
    # Repeat for each label (e.g. C0013264, C0917713, C3542021)
    X_new_df = pd.DataFrame(tokens, columns=['token'])
    # print(X_new_df)
    for label in range(len(labelsNames)):
        # Fit scores per feature
        selector_scores = selector.fit(X, Y[labelsNames[label]])
        # print(selector_scores.scores_ )
        # Add scores in the dataframe (as a pd.Series)
        X_new_df.loc[:,labelsNames[label]] = pd.Series(selector_scores.scores_)
    X_new_df.loc[:,MaxWeights] = X_new_df[labelsNames].max(axis=1)
    # X_new_df[MaxWeights] = X_new_df[labelsNames].mean(axis=1)
    return X_new_df

def getTopFeatures(tfidf, X_df, count, tokens, kFeatures):
    '''
    Transform the dataset (i.e. :param tfidf:, :param count: and :param tokens: variables) to include only the top :param kFeatures: features
        Performs Feature selection using the scores in X_df, which ~!~!~!~ must be sorted descending by Max Weight across all labels ~!~!~!~

    :param tfidf:           A scipy.sparse.csr_matrix (document id, token id) with TFIDF values
    :param X_df:            A pandas.DataFrame with size (features, labels+2) the weights for each feature:
                                - One column for each label (e.g. C0013264, C0917713, C3542021)
                                - One column with the tokens ('token') and
                                - One column with the maximum weight for each token across all labels
                                (*) The dataframe should be sorted descending by Max Weight across all labels
    :param count:           A scipy.sparse.csr_matrix (document id, token id) with token counts
    :param tokens:          A pandas.DataFrame with columns : 'token'. Provides a mapping from token id to token string
    :param kFeatures:       The number of top features to be kept in the dataset
    :return:    The transformed dataset considering only the top :param kFeatures: features selected by feature selection
        1) tfidf_selectedFeatures   A scipy.sparse.csr_matrix (document id, token id) considering only the :param kFeatures: features selected
        2) count_selectedFeatures   A scipy.sparse.csr_matrix (document id, token id) considering only the :param kFeatures: features selected
        3) tokens_selectedFeatures  A pandas.DataFrame with columns : 'token'. Provides a mapping from (new) token id to token string only for the :param kFeatures: features selected
    '''

    # Select top features/token ids only
    # Create a mask list with True for the selected features (i.e. top n)
    featureSelectionMask = [True] * kFeatures + [False] * (len(tokens) - kFeatures)
    # Add the mask column in the DataFrame
    X_df.loc[:,'FeatureMask'] = featureSelectionMask
    # Get the ids of the selected features
    featuresSelected = X_df.index[featureSelectionMask].tolist()
    # print(featuresSelected)

    # Transofm tfidf matrix (keep selected features only)
    tfidf_selectedFeatures = sparse.lil_matrix(sparse.csr_matrix(tfidf)[:, featuresSelected])
    tfidf_selectedFeatures = scipy.sparse.csr_matrix(tfidf_selectedFeatures)
    # print(tfidf_selectedFeatures.shape)

    # Transform count matrix (keep selected features only)
    count_selectedFeatures = sparse.lil_matrix(sparse.csr_matrix(count)[:, featuresSelected])
    count_selectedFeatures = scipy.sparse.csr_matrix(count_selectedFeatures)
    # print(count_selectedFeatures.shape)

    # Transform tokens DataFrame (keep selected features only)
    tokens_selectedFeatures = tokens.iloc[featuresSelected]
    # Selected Token/term strings (in the order appearing in tfidf and count matrices)
    new_tokens = list(tokens_selectedFeatures['token'])
    # Create new index
    new_index = range(len(featuresSelected))
    tokens_selectedFeatures = pd.DataFrame(new_tokens, index=new_index, columns={'token'})

    return tfidf_selectedFeatures, count_selectedFeatures, tokens_selectedFeatures

