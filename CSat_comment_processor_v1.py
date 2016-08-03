import nltk, re, math, collections, lda, random, json, wordcloud, sys, io
import numpy as np
import pandas as pd
from matplotlib import pyplot
import time
from sklearn.cluster import KMeans
from nltk.util import ngrams

FILENAME = "comments_raw_merged_with_page_data_190716.csv"

LANGUAGE = "english"
STOPWORD_SET = (set(nltk.corpus.stopwords.words(LANGUAGE)) | set(['even', '.', ','])) - set(['who', 'why', 'how', 'where', 'when', 'what', 'whom']) 
COLUMN_NAME = "message"
LDA_TOPIC_NUMBER = 3 # Kind of trialed and errored.
LDA_ITER = 500 # Don't change this!
LDA_RANDOM_STATE = 1 # Don't change this!
SEED_NUMBER = 10
THANK_YOU_WORD = 'thank'
THANKFUL_CATEGORY = 1
POST_ID_COLUMN_NAME = 'post_id'
TF_WORD_THRESHOLD = 5
MAX_ARTICLE_LIMIT = 3
ARTICLE_ALERT_SCORE_CUTOFF = 0.7
MAX_COMMENT_LIMIT = 3
COMMENT_SCORE_CUTOFF = 0.1
URL_PATH_COLUMN_NAME = "page_path_tx"
MIN_CLUSTER_COUNT = -1
KEYWORDS_PER_CLUSTER = 5
WORDCLOUD_MAX_WORDS = 100
WORDCLOUD_HEIGHT = 400
WORDCLOUD_WIDTH = 800
COMMENTS_PER_POSTS_RATIO = 100 
MIN_COMMENTS_PER_POST = 10
AVERAGE_CSAT_SCORE = 0.5
MAX_KEYWORD_CLUSTERS = 3

def is_ASCII_string(string):
	'''
		Checks if string is ASCII-decodeable. Always run on pre-processed csv files!
	'''
	try:
		string.decode('ascii')
		return True
	except:
		return False


def ascii_substituted(string):
	'''
		Substitutes string to make it ASCII-readable. Currently very exception-driven, 
		but if there's a good package that does the dirty work please let @Lumpy know.
	'''
	return string.replace("\\n", " ").replace("&amp;", "&").replace('&#039;', '\'').replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">").replace("\xe2\x80\x93", "-").replace("\xe2\x80\x99", "\'").strip()

### CLEAN YOUR DF!!
def get_message_list(data_frame, column_name = "message"):
	'''
		Given a pre-CLEANED and pre-subsetted data frame, gets the message set from the data frame.
	'''
	return [message for message_l in data_frame[[column_name]].values.tolist() for message in message_l]            


def is_eligible_word(token, stopwords = STOPWORD_SET, regex_string = "^[^a-zA-Z0-9]+"):
	'''
		Private helper function.
		Check if word is eligible to be a token (i.e. not a forbidden regex, or in the stopword set).
	'''
	pattern = re.compile(regex_string)
	if pattern.match(token) or token in stopwords: 
		return False
	return True


def append_NOTs(tokenized_message, stopwords = STOPWORD_SET):
	'''
		Private helper function to handle negations.
		Given a List of words, returns a List of words with nots appended to the right words.
	'''
	new_message = []
	for i in range(len(tokenized_message)):
		if tokenized_message[i] in set(["n\'t", "n\"t", "no", "not", "didnt"]) and i != (len(tokenized_message) - 1):
			j = i + 1
			while j < len(tokenized_message):
				if tokenized_message[j] not in stopwords:
					tokenized_message[j] = "not_" + tokenized_message[j]
					break
				j += 1
		else:
			new_message.append(tokenized_message[i])
	return new_message


def split_message_into_tokens(untokenized_message, ngram = 1, stopwords = STOPWORD_SET, regex_string = "^[^a-zA-Z0-9]+", handle_negations = True): 
	'''
		Splits ONE message in the list of messages into words/tokens, in the process doing the following:
		1) Changing to lowercase
		2) Dealing with negations
		3) N-gramming
		4) Lemmatizing
		5) Stopword-removal
		Input: String.
		Arguments: 
			ngram:              n in n-grams                        Default is 1
			stopwords           Set of stopwords to remove          Default is the global variable
			regex_string        String containing regex pattern     Default is "^[^a-zA-Z0-9]+"
			handle_negations:   Do you want negations handled?      Default is True
		Output: List of n-grams.

		Note: A PorterStemmer doesn't work very well, because the lemmatizer often is unable to re-stem many words. A better method would be just to put a lemmatizer, with default POS "v" for verb.
	'''
	wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
	#stemmer = nltk.stem.porter.PorterStemmer()
	# Split into tokens
	tokenized_message = nltk.word_tokenize((ascii_substituted(str(untokenized_message))).lower())
	# Deal with "not"s
	if handle_negations:
		tokenized_message = append_NOTs(tokenized_message, stopwords)
	# Lemmatize, n-gram, stopword-removal.
	if ngram == 1:
		return [wordnet_lemmatizer.lemmatize(token, "v") for token in tokenized_message if is_eligible_word(token, stopwords, regex_string)]
	else:
		tokenized_message = ngrams(tokenized_message, ngram)
		filtered_tokenized_message = [" ".join(str(wordnet_lemmatizer.lemmatize(token, "v")) for token in ngram_indiv if is_eligible_word(token, stopwords, regex_string)) for ngram_indiv in tokenized_message]
		return [final_token for final_token in filtered_tokenized_message if len(final_token) > 0] 


def tokenize(message_list, ngram = 1, stopwords = STOPWORD_SET, regex_string = "^[^?a-zA-Z0-9]+", handle_negations = True, pos_tagset = "no-pos"):
	'''
		Tokenizes a list of strings. Assumes that message_list has been preprocessed.
		Input: List of Strings.
		Arguments:
			ngram               n in n-grams                                                                                Default is 1
			stopwords:          Set of stopwords to remove.                                                                 Default is the global variable.
			regex_string        String containing regex pattern                                                             Default is "^[^a-zA-Z0-9]+"
			handle_negations:   Do you want "not X" to be glued as "not_X"? Improves sentiment recognition.                 Default is True
			pos_tagset:         For POS tagging. Current options are "universal", None (which maps to nltk.pos_tag) 
								and "no-pos", which really means no POS tagging. Only applicable for unigrams.              Default is None
		Output: List of List of Strings.
		Note: will automatically change to lowercase, and will automatically lemmatize.
	'''
	if ngram == 1 and (pos_tagset == None or pos_tagset == "universal"): ## Will perform POS-tagging
		return [nltk.pos_tag(split_message_into_tokens(untokenized_message, ngram = ngram, stopwords = stopwords, regex_string = regex_string, handle_negations = handle_negations), pos_tagset = pos_tagset) for untokenized_message in message_list]
	else:
		return [split_message_into_tokens(untokenized_message, ngram = ngram, stopwords = stopwords, regex_string = regex_string, handle_negations = handle_negations) for untokenized_message in message_list]


def get_unnested_list(word_list):
	'''
		Gets an un-nested list from a nested list of words. Deals in principle with ngrams with n > 1.
	'''
	if len(word_list) == 0 or (type(word_list[0]) not in set([list, dict, set, tuple])):
		return word_list
	else:
		return get_unnested_list([word for word_array in word_list for word in word_array])

	
def get_token_to_index_dict(token_list, threshold = TF_WORD_THRESHOLD):
	'''
		Gets a dict mapping tokens to indices, given an (unnested) list of tokens.
	'''
	token_list = sorted([item[0] for item in collections.Counter(token_list).items() if item[1] >= threshold])
	return {token_list[i]:i for i in range(len(token_list))}

	
def get_ordered_array_of_tokens(token_to_index_dict):
	''' 
		Gets back the ordered array of tokens, with the index for each token being the value in the dict.
		Obviously assumes that the dict's values are all unique and range from 0 to len(dict) - 1. 
	'''
	return [token_index_tuple[0] for token_index_tuple in sorted(token_to_index_dict.items(), key = lambda x: x[1])]


def get_TF_matrix(tokenized_message_list, token_to_index_dict, count_only_once = set([])):
	## FIXED: NOW matrix is n, p dimensions where n is number of training documents and p rightfully is predictor (word)
	''' 
		Returns a NumPy Matrix where the (i, j)th position represents the number of times word j is present in document i. 
		Input: 
			tokenized_message_list:     List of List of tokens.
			token_to_index_dict:        Dict mapping token to index
		Arguments:
			count_only_once:            Set of tokens to only count once for each document.         Default: empty set
		Output: Term-Frequency Matrix    
	'''
	n_documents = len(tokenized_message_list)
	n_words = len(token_to_index_dict)
	tf_matrix = np.zeros((n_documents, n_words), dtype=np.int)
	for i in range(n_documents):
		tokenized_message = tokenized_message_list[i]
		for token in tokenized_message:
			if token in token_to_index_dict:
				j = token_to_index_dict[token]
				if token in count_only_once:
					tf_matrix[i, j] = 1
				else:
					tf_matrix[i, j] += 1
	return tf_matrix

	
def get_IDF_score(tf_matrix):
	'''
		Given the Term-Frequency Matrix, counts the inverse-document-frequency score for the term.
		Calculated as IDF (token) = ln( N / (number of docs token is in + 1)),
		where N is number of document, and the +1 is for smoothing purposes.
		Returns a List of floats.
	'''
	idf_score = []
	n_documents = tf_matrix.shape[0]
	for j in range(tf_matrix.shape[1]):
		n_appearing_docs = len(np.where(tf_matrix[:,j] > 0)[0])
		idf = math.log(float(n_documents) / (n_appearing_docs) + 1)
		idf_score.append(idf)
	return idf_score


def get_TFIDF_matrix(tf_matrix, idf_score, multiple_counts = True, normalized = True):
	## Assume L2 normalization. I'm kinda lazy to implement L1
	## MultipleCounts = False means all nonzero iterms in tf_matrix becomes 1
	## Add functionality to only singularcount a subset of words
	'''
		Given the TF matrix and the IDF score, calculates the TF-IDF matrix.
		Inputs: 
			tf_matrix: A (n-documents * p-words) matrix.
	'''
	n_rows, n_cols = tf_matrix.shape
	tf_idf = np.zeros(tf_matrix.shape)
	for i in range(n_rows):
		for j in range(n_cols):
			tf_idf[i, j] = float(tf_matrix[i, j] if multiple_counts else min(tf_matrix[i, j], 1)) * idf_score[j]
	if normalized:
		tf_idf = normalize(tf_idf, axis = 1, norm = 'l2')
	return tf_idf


#### If you want to rework the algorithm, look here
def get_LDA_scores(tf_matrix, token_to_index_dict, seed_no = SEED_NUMBER, n_topics = LDA_TOPIC_NUMBER, n_iter = LDA_ITER, lda_random_state = LDA_RANDOM_STATE):
	'''
		Computes individual comment scores via LDA, as well as returns the index for which sentiment was positive
	'''
	random.seed(seed_no)
	model = lda.LDA(n_topics = LDA_TOPIC_NUMBER, n_iter = LDA_ITER, random_state = LDA_RANDOM_STATE)
	model.fit(tf_matrix)
	## Detects the one with the "thank you"s. If "thank" doesn't exist, it will go berserk - but that's the point!
	topic_word = model.topic_word_
	if THANK_YOU_WORD in token_to_index_dict:
		index_of_positive_sentiment_topic = np.argsort(model.topic_word_[:, token_to_index_dict[THANK_YOU_WORD]])[:1:-1][0]
	else:
		index_of_positive_sentiment_topic = THANKFUL_CATEGORY ### SOUND OFF
		print("\'thank\'' was not found. Use with caution.")
	return np.dot(tf_matrix, np.transpose(model.components_)), index_of_positive_sentiment_topic # this gives you the full topic scores for each item in the DF
#### Edit a bit of the next function too.


def compute_final_scores(raw_comment_scores, LDA_positive_category):
	'''
		Given the TF matrix (optionally tf_idf) and the token_to_index_dict:
		Gets you back comment scores for all comments. Use them to rank how high articles should go.
		PASS 1: Vanilla LDA, implemented 28 July 2016. Calls the get_LDA_scores function
	'''
	if LDA_positive_category < 0 or LDA_positive_category >= raw_comment_scores.shape[1]:
		return np.sum(raw_comment_scores, axis = 1)
	return np.sum(np.delete(raw_comment_scores, LDA_positive_category, axis = 1), axis = 1)
	 

def get_posts_scores(df, raw_comment_scores, LDA_positive_category, column_to_group_by = POST_ID_COLUMN_NAME):
	'''
		Given the data frame, raw comment scores and the positive LDA category, gets back a numpy matrix with post_id and post scores, sorted by the biggest to smallest post score (bigger = worse).
	'''
	final_comment_scores = compute_final_scores(raw_comment_scores, LDA_positive_category)
	posts = df.groupby(column_to_group_by)
	final_post_scores = np.zeros((len(posts.indices), 2))
	index = 0
	for post_id, comment_indices in posts.indices.iteritems():
		final_post_scores[index, 0] = post_id
		final_post_scores[index, 1] = np.sum(final_comment_scores[comment_indices])
		index += 1
	return final_post_scores[final_post_scores[:, 1].argsort()[::-1]]

#### get_posts_scores_v2 is an attempt to work on the current get_posts_scores algorithm. Read more in the docstring
def get_posts_scores_v2(df, raw_comment_scores, LDA_positive_category, column_to_group_by = POST_ID_COLUMN_NAME):
	'''
		This attempt will calculate the posts score based on the formula above:
		Post score = [sum (bad_quality_score) / (sum (bad_quality_score) + sum (good_quality_score))]. 
		Some tweaks will have to be done to factor in the number of comments.
		There will be a couple of comments in the raw_comment_scores with 0 scores across the board (consider them as "junk comments"). Just make sure you know that.       
	'''
	posts = df.groupby(column_to_group_by)
	final_post_scores = np.zeros((len(posts.indices), 2))
	index = 0
	for post_id, comment_indices in posts.indices.iteritems():
		final_post_scores[index, 0] = post_id
		post_good_and_bad_score = np.sum(raw_comment_scores[comment_indices, :])
		csat_score = calculate_csat_scores(df.iloc[comment_indices])
		if post_good_and_bad_score > 0 and len(comment_indices) > max(MIN_COMMENTS_PER_POST, (raw_comment_scores.shape[0] / COMMENTS_PER_POSTS_RATIO)) and csat_score < AVERAGE_CSAT_SCORE:
			final_post_scores[index, 1] = np.true_divide(np.sum(np.delete(raw_comment_scores[comment_indices, :], LDA_positive_category, axis = 1)), np.sum(raw_comment_scores[comment_indices, :]))
		else:
			final_post_scores[index, 1] = 0
		index += 1
	return final_post_scores[final_post_scores[:, 1].argsort()[::-1]]


''' Kind of depreciated for now, but if ever it comes useful...
def get_posts_scores_csat(df):
	posts = df.groupby(POST_ID_COLUMN_NAME)
	csat_scores = np.zeros((len(posts.indices), 2))
	index = 0
	for post_id, comment_indices in posts.indices.iteritems():
		csat_scores[index, 0] = post_id
		csat_scores[index, 1] = calculate_csat_scores(df.iloc[comment_indices]) if len(comment_indices) > 10 else None
		index += 1
	final_post_scores = csat_scores[csat_scores[:, 1].argsort()]
	return final_post_scores[~np.isnan(final_post_scores[:, 1])]
'''

def calculate_csat_scores(df):
	'''
		A function to get back the CSat scores for a particular post_id, given the subsetted data frame.
	'''
	counts = df["helpful_yn"].value_counts()
	df_nrow = df.shape[0]
	if df_nrow > 0 and "yes" in counts.index:
		return float(counts[np.where(counts.index == "yes")[0][0]]) / df_nrow
	else:
		return 0 # kind of a catch-all. may change it to -1 at some point


def find_index_of_positive_sentiment_topic(LDA_scores, token_to_index_dict, thank_you_word = THANK_YOU_WORD):
	'''
		In anticipation of future expansion:
		Given a token_to_index dict, and the saved LDA scores, manually find the index of the topic that embodies positive sentiment.
		Will be extremely useful in grading scores.
	'''
	assert LDA_scores.shape[0] == len(token_to_index_dict)
	return np.argsort(LDA_scores[:, token_to_index_dict[thank_you_word]])[:1:-1][0]



## TO FIX
def obtain_frequent_keywords(indices_of_comments, raw_comment_scores, tf_matrix, idf_score, token_to_index_dict, min_cluster_count = MIN_CLUSTER_COUNT, seed_number = SEED_NUMBER, KEYWORDS_PER_CLUSTER = KEYWORDS_PER_CLUSTER):
	'''
		First cluster into k different clusters based on LDA score.
		Next, pick out the top few clusters (assumption, same words/similar words will have the same kind of scores.)
		Within the top few clusters, pick out the top-most TF-IDF scores, assuming you put it all together. 
		
		Inputs:
			indices_of_comments: the DF indices of comments, so as to access the tf_matrix.
			raw_comment_scores: the np.array of LDA comment scores (in full), so then indices_of_comments can subset the relevant ones
			tf_matrix: the Term Frequency matrix. (n * p, where n should be the same DF indices, and p the number of words)
			idf_score: idf score for the DF in full
			token_to_index_dict: the dictionary mapping out token to index
	'''
	relevant_comments_normalized_scores = np.true_divide(raw_comment_scores[indices_of_comments], np.sum(raw_comment_scores[indices_of_comments], axis = 1)[:, None])
	# clearing out the 0s
	indices_of_comments = indices_of_comments[~np.all(np.isnan(relevant_comments_normalized_scores), axis = 1)]
	relevant_comments_normalized_scores = relevant_comments_normalized_scores[~np.all(np.isnan(relevant_comments_normalized_scores), axis = 1)]
	k = int(math.floor(math.sqrt(len(indices_of_comments)))) # k = floor(sqrt(# of comments))
	if min_cluster_count < 0:
		min_cluster_count = k
	if seed_number < 0: # there's a way to disable seeding. Why would you not want seeds though? 
		random.seed(seed_number)
	km = KMeans(n_clusters = k, n_init = LDA_ITER)
	km.fit(relevant_comments_normalized_scores)
	cluster_counts = collections.Counter(km.labels_).most_common(k)
	frequent_clusters = [count[0] for count in cluster_counts if (count[1] > min_cluster_count or count[1] >= cluster_counts[0][1])]
	frequent_clusters = frequent_clusters[0:min(len(frequent_clusters), MAX_KEYWORD_CLUSTERS)]
	array_of_tokens = get_ordered_array_of_tokens(token_to_index_dict)
	frequent_keywords = {}
	topic_number = 1   
	for cluster in frequent_clusters:
		cluster_indices = indices_of_comments[np.where(np.array(km.labels_) == cluster)]
		tf_sum = np.sum(tf_matrix[cluster_indices, :], axis = 0)
		tf_idf_scores = tf_sum * np.array(idf_score)
		cluster_name = "Keywords of Topic " + str(topic_number)
		topic_number += 1
		frequent_keywords[cluster_name] = [array_of_tokens[i] for i in (tf_idf_scores.argsort()[::-1][0:KEYWORDS_PER_CLUSTER]).tolist()]
	return frequent_keywords



def get_articles(df, final_post_scores, raw_comment_scores, vertical_name, tf_matrix, idf_score, token_to_index_dict, max_article_limit = MAX_ARTICLE_LIMIT, max_comment_limit = MAX_COMMENT_LIMIT, ARTICLE_ALERT_SCORE_CUTOFF = ARTICLE_ALERT_SCORE_CUTOFF, comment_score_cutoff = COMMENT_SCORE_CUTOFF):
	posts = df.groupby(POST_ID_COLUMN_NAME)
	important_comment_scores = compute_final_scores(raw_comment_scores, -1) # length - full df
	final_category_output = []
	for i in range(min(max_article_limit, final_post_scores.shape[0])):
		if final_post_scores[i, 1] <= ARTICLE_ALERT_SCORE_CUTOFF:
			break
		post_id = int(final_post_scores[i, 0])
		post_alert_score = final_post_scores[i, 1]
		indices_of_comments = posts.indices[post_id] 
		post_url = df[URL_PATH_COLUMN_NAME].iloc[indices_of_comments[0]] # any comment will do. They share the same URL anyway.
		post_csat_score = calculate_csat_scores(df.iloc[indices_of_comments])
		comment_count = len(indices_of_comments) 
		post_info = {"post_id": post_id, "post_url": post_url, "post_alert_score": post_alert_score, "post_csat_score": post_csat_score, "comment_count": comment_count}
		comment_array = []
		applicable_comment_scores = np.transpose(np.vstack((np.array(indices_of_comments), np.array(important_comment_scores[indices_of_comments])))) # np array with df index, and its comment score
		applicable_comment_scores = applicable_comment_scores[applicable_comment_scores[:, 1].argsort()[::-1]] # these should be index of df
		for j in range(min(max_comment_limit, applicable_comment_scores.shape[0])): 
			if applicable_comment_scores[j, 1] <= comment_score_cutoff:
				break
			comment_index = int(applicable_comment_scores[j, 0])
			log_id = df["log_id"].iloc[comment_index]
			comment_text = ascii_substituted(str(df[COLUMN_NAME].iloc[comment_index]))
			comment_array.append({"log_id": log_id, "comment_text": comment_text})
		post_info["comment"] = comment_array
		post_info["keywords"] = obtain_frequent_keywords(indices_of_comments, raw_comment_scores, tf_matrix, idf_score, token_to_index_dict)
		final_category_output.append(post_info)
	return {"vertical": vertical_name, "summary": final_category_output}


'''
An example of what the JSON format would be like:
{	"vertical": "Credit Cards",
	"summary": [
		{	"post_id": "538", 
			"post_csat_score": 0.1, 
			"comment_count": 100,
			"post_alert_score": 0.802,
			"post_url": "/blog/make-donald-drumpf-again", 
			"comment": [
				{	"log_id": 24601,
					"comment_text": "father father father help us"
				}
					]}, 
}
'''


##### WORDCLOUD
def generate_wordcloud(df, vertical_name, ngram = 2, regex_string = "^[^a-zA-Z0-9]+", height = WORDCLOUD_HEIGHT, width = WORDCLOUD_WIDTH):
	'''
		Generates the wordcloud. 
		Input: Pandas dataframe
		Output: A wordcloud, saved in whatever relevant drive
	'''
	# Step 1: Generate token count.
		# (ngram = 2 works. More than that doesn't add more)
	tokenized_positive = get_unnested_list(tokenize(get_message_list(df[df['helpful_yn'] == "yes"]), ngram = ngram, regex_string = regex_string))
	token_counts_positive = pd.DataFrame(collections.Counter(tokenized_positive).items())
	token_counts_positive["Percentage"] = token_counts_positive[1] / token_counts_positive[1].sum()
	token_counts_positive.rename(columns = {0:"Bigram", 1:"Counts"}, inplace = True)
	tokenized_negative = get_unnested_list(tokenize(get_message_list(df[df['helpful_yn'] == "no"]), ngram = ngram, regex_string = regex_string))
	token_counts_negative = pd.DataFrame(collections.Counter(tokenized_negative).items())
	token_counts_negative["Percentage"] = token_counts_negative[1] / token_counts_negative[1].sum()
	token_counts_negative.rename(columns = {0:"Bigram", 1:"Counts"}, inplace = True)

	# Step 2: Calculate percent difference between words in a "helpful CSat" vs an "unhelpful CSat". 
	token_counts = token_counts_positive.merge(token_counts_negative, how = "outer", on = "Bigram").fillna(0)
	token_counts["Difference"] = token_counts["Percentage_x"] - token_counts["Percentage_y"]
	## DO WE WANT TO ADD POS tag? An option can be done with the following code:
	# token_counts["POS"] = token_counts["Bigram"].apply(lambda x: nltk.pos_tag([x], tagset = "universal")[0][1]) 
	# No simple way to deal with POS tagging, because POS tagger is not very good especially for common words. 
	# Potential fix: weigh scores with IDF scores.
		# Complication: it seems like the WordCloud package already deals with it in some random form...to investigate

	# Step 3: Generate wordcloud based on that, and save to file
	wc = wordcloud.WordCloud(height = height, width = width)
	wc.generate_from_frequencies(token_counts[["Bigram", "Difference"]].sort_values(by = "Difference", ascending = False).iloc[0:WORDCLOUD_MAX_WORDS].values.tolist())
	wc.to_file(vertical_name + "_positive_wordcloud.png")
	wc = wordcloud.WordCloud(height = height, width = width)
	wc.generate_from_frequencies(token_counts[["Bigram", "Difference"]].sort_values(by = "Difference").iloc[0:WORDCLOUD_MAX_WORDS].values.tolist())
	wc.to_file(vertical_name + "_negative_wordcloud.png")


## STEP 1: PREPROCESSING
df = pd.read_csv(FILENAME, delimiter = ',')
faulty_arrays = []
good_arrays = []

messageList = df["message"].values.tolist()

for i in range(len(messageList)):
	message = messageList[i]
	if not is_ASCII_string(ascii_substituted(str(message))):
		faulty_arrays.append(i)
	else:
		good_arrays.append(i)
		
df = df.iloc[good_arrays, :]


## STEP 2: Getting it in a JSON format:
def comment_extractor(df, vertical_name, ngram = 2, threshold = TF_WORD_THRESHOLD, ARTICLE_ALERT_SCORE_CUTOFF = ARTICLE_ALERT_SCORE_CUTOFF, comment_score_cutoff = COMMENT_SCORE_CUTOFF):
	'''
		The important function. Comment Extraction is the mainstay of this package, isn't it?
	'''
	tokenized_message_list = tokenize(get_message_list(df), ngram = ngram)
	token_to_index_dict = get_token_to_index_dict(get_unnested_list(tokenized_message_list), threshold = threshold)
	tf_matrix = get_TF_matrix(tokenized_message_list, token_to_index_dict)
	idf_score = get_IDF_score(tf_matrix)
	raw_comment_scores, LDA_positive_category = get_LDA_scores(tf_matrix, token_to_index_dict)
	final_comment_scores = compute_final_scores(raw_comment_scores, LDA_positive_category)
	final_post_scores = get_posts_scores_v2(new_df, raw_comment_scores, LDA_positive_category)
	return get_articles(new_df, final_post_scores, raw_comment_scores, category, tf_matrix, idf_score, token_to_index_dict, comment_score_cutoff = comment_score_cutoff, ARTICLE_ALERT_SCORE_CUTOFF = ARTICLE_ALERT_SCORE_CUTOFF)

json_printout = []
for category in df["page_vertical_tx"].unique().tolist():
	new_df = df[df["page_vertical_tx"] == category]
	if new_df.shape[0] > 0:
		extract = comment_extractor(new_df, category)
		print(json.dumps(extract, indent = 4))
		json_printout.append(extract)


def main():
	try:
		file = sys.argv[1]
		print ("Opening file: " + file)
		df = pd.read_csv(file, delimiter = ',')
	except IOError:
		print ("Filename Error. Reverting to default filename: " + FILENAME)
		print ("Opening file: " + FILENAME)
		df = pd.read_csv(file, delimiter = ',')
	print("Preprocessing data frame...")
	faulty_arrays = []
	good_arrays = []
	messageList = df["message"].values.tolist()
	for i in range(len(messageList)):
		message = messageList[i]
		if not is_ASCII_string(ascii_substituted(str(message))):
			faulty_arrays.append(i)
		else:
			good_arrays.append(i)
	df = df.iloc[good_arrays, :]
	print("Done. Extracting comments...")
	json_printout = {}
	for category in df["page_vertical_tx"].unique().tolist():
		new_df = df[df["page_vertical_tx"] == category]
		if new_df.shape[0] > 0:
			extract = comment_extractor(new_df, category)
			#print(json.dumps(extract, indent = 4))
			json_printout[category] = extract
	output_file_name = "comment-feed.txt" if len(sys.argv) < 3 else sys.argv[2]
	with open(output_file_name, "w") as file:
		json.dump(json_printout, file)


if __name__ == "__main__":
	main()




'''
Procedure:

1) Most Urgent Articles:
Within a vertical, gather all comments within a certain timeframe.
Each comment will have a "negative comment" score, and a "positive comment" score. This is calculated via LDA (more negative words in there, worse it gets.)

- however when scoring, we train based on a full LDA score (i.e. use the trained-on-full-vertical LDA topic - word matrix)
--- problem with this is that the trained-on-full-vertical LDA topic-word matrix is sometimes p bad for small verticals.
--- To calculate "goodness", train on full CSat LDA matrix? (pre-tune. So what happens is I save the matrix on a file, and use that).

Alternatiely, we could look at CSat Scores. (let's try that?)

2) Best comments:

3) Most commonly seen requests/most discussed topics:

'''

'''
Sprint, week 7:

Design and implement program to:
1. Automate the pulling of DWH data, keeping track of of weekly updates 
2. Incorporate automated feedback from AEs
Output: updated Python + whatever else, to dump into Git repo

1) Missing 0-1 scale for article's ALERT score 
FIXED! Sort by ALERT score, but only showcase the real CSat score.
FIXED! Put a fractional limit.
FIXED! Will showcase the real CSat Score, instead of the ALERT score (that is pretty bs)

2) Terrible "frequent topic detection" - TF-IDF approach + closest-to-centroid approaches tried and not particularly successful. 
- Fix LDA matrix to manually increase weights of particular keywords (question words) - whatever for? to increase the types of questions coming back.
4) Penalize bad spelling more than current algorithm (and do a spellcheck)
- how? increase threshold first, see how it works. Seems to do good enough. 

Each time I pull from the DWH:
1) I pull the whole damn thing, and therefore track by log_id
2) As a CSV:
	- Track every single log_id since eternity (?), as well as a Good-Bad-Never_reviewed trichotomy
	- Each time it comes back as reviewed, update the table.
	- (which means any form of data has to be transmitted via log_id)
3) Save these things:
	- the list of terms (best in a dict; if not also can be just a list of tokens, separated by tab/newline)
	- the LDA table
	- the POSITIVE CATEGORY group (alternatively I could write a simple function to find it.)
4) 

Next thing to work on: clean up the global vars! messy as hell

'''
