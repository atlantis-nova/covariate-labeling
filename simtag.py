from scipy.linalg import lstsq, solve, LinAlgWarning
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter


def _nnls(A, b, maxiter=None, tol=None):
	"""
	This is a single RHS algorithm from ref [2] above. For multiple RHS
	support, the algorithm is given in  :doi:`10.1002/cem.889`
	"""
	# with warnings.catch_warnings():
	warnings.filterwarnings("ignore", category=LinAlgWarning)

	m, n = A.shape

	AtA = A.T @ A
	Atb = b @ A  # Result is 1D - let NumPy figure it out

	if not maxiter:
		maxiter = 3*n
	if tol is None:
		tol = 10 * max(m, n) * np.spacing(1.)

	# Initialize vars
	x = np.zeros(n, dtype=np.float64)
	s = np.zeros(n, dtype=np.float64)
	# Inactive constraint switches
	P = np.zeros(n, dtype=bool)

	# Projected residual
	w = Atb.copy().astype(np.float64)  # x=0. Skip (-AtA @ x) term

	# Overall iteration counter
	# Outer loop is not counted, inner iter is counted across outer spins
	iter = 0

	while (not P.all()) and (w[~P] > tol).any():  # B
		# Get the "most" active coeff index and move to inactive set
		k = np.argmax(w * (~P))  # B.2
		P[k] = True  # B.3

		# Iteration solution
		s[:] = 0.
		# B.4
		with warnings.catch_warnings():
			warnings.filterwarnings('ignore', message='Ill-conditioned matrix', category=LinAlgWarning)
			try:
				s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym', check_finite=False)
			except:
				s[P] = lstsq(AtA[np.ix_(P, P)], Atb[P], check_finite=False)[0]
		
		# Inner loop
		while (iter < maxiter) and (s[P].min() < 0):  # C.1
			iter += 1
			inds = P * (s < 0)
			alpha = (x[inds] / (x[inds] - s[inds])).min()  # C.2
			x *= (1 - alpha)
			x += alpha*s
			P[x <= tol] = False
			with warnings.catch_warnings():
				warnings.filterwarnings('ignore', message='Ill-conditioned matrix', category=LinAlgWarning)
				try:
					s[P] = solve(AtA[np.ix_(P, P)], Atb[P], assume_a='sym', check_finite=False)
				except:
					s[P] = lstsq(AtA[np.ix_(P, P)], Atb[P], check_finite=False)[0]
			s[~P] = 0  # C.6

		x[:] = s[:]
		w[:] = Atb - AtA @ x

		if iter == maxiter:
			# Typically following line should return
			# return x, np.linalg.norm(A@x - b), -1
			# however at the top level, -1 raises an exception wasting norm
			# Instead return dummy number 0.
			return x, 0., -1

	return x, np.linalg.norm(A@x - b), 1


def covariate_labeling(tag_list, tag_encoded, vector, inflection_point):

	def pairwise_slices(lst):
		ranges = [lst[i:i+2] for i in range(len(lst)-1)]
		return ranges

	try:
		X = np.hstack([x[:, np.newaxis] for x in tag_encoded])
		# Non-Negative Least Squares (NNLS)
		# w, _ = nnls(X, vector, maxiter=1000)
		w, rnorm, info = _nnls(X, vector, maxiter=100)
		# print(w, _)

		# filtering
		data = sorted(list(zip(w, tag_list)))[::-1]
		zipped = [x for x in data[::-1] if x[0] > 0]
		df_ = pd.DataFrame(zipped, columns=['Score', 'Label'])
		scores = df_['Score'].values.tolist()
		# print(scores)

		# Calculate the first and second derivatives
		first_derivatives = np.diff(scores)
		second_derivatives = np.diff(first_derivatives)

		# Calculate the absolute values of the second derivatives
		abs_second_derivatives = np.abs(second_derivatives)
		try:
			inflection_points = sorted(np.argsort(-abs_second_derivatives)[:3])[::-1]
			tags_top = df_.iloc[df_[inflection_points[inflection_point]:].index]['Label'].tolist()
		except:
			tags_top = [x[1] for x in zipped]
	
	except:
		return 0, 0
		
	lists = list()
	lst = [len(zipped)]+inflection_points+[0]
	lst = list(set(lst))[::-1]
	ranges = pairwise_slices(lst)
	counter = len(inflection_points)
	for r in ranges:
		[lists.append(x) for x in [counter]*abs(r[1]-r[0])]
		counter -= 1
	
	scores = [[zipped[x][0], lists[::-1][x], zipped[x][1]] for x in range(len(zipped))][::-1]
	
	# # print z-score
	# fig = px.bar(df_[::-1], x=df_.index, y='Score')
	# fig.show()
	return df_['Score'].values.tolist(), tags_top, scores


def knn_comparison(index, vector, tag_list, tags_top):
	# compare it with raw knn: the positions are scattered, we don't just keep the top k ones
	distances, indices = index.kneighbors([vector])
	knn_tags = list()
	for i in indices[0]:
		if tag_list[i] in tags_top:
			knn_tags.append(1)
		else:
			knn_tags.append(0)
	return knn_tags


def downgrade_tag(scores, tag_name):
	"""
	Downgrades one tag to the last position living the scoring intact
	"""
	index = [x[2] for x in scores].index(tag_name)
	scores[index][0] = min([x[0] for x in scores]) - scores[index][0]
	ip = [x[1] for x in scores]
	scores = sorted(scores)[::-1]
	scores = [[scores[x][0], ip[x], scores[x][2]] for x in range(len(scores))]
	return scores


def count_df(df):
	results_tag_list = list()
	for x in df['_tags'].tolist():
		results_tag_list += x
	results_tag_list = [x[2] for x in results_tag_list if x[1]>1]
	counter = Counter(results_tag_list)
	counter = list(counter.items())
	counter = sorted([[x[1], x[0]] for x in counter])[::-1]
	return counter


def plot_df(df, max_categories=1000):
	# Convert the list to a pandas DataFrame
	df_plot = pd.DataFrame(count_df(df), columns=['Value', 'Category'])[0:max_categories]
	df_plot['Category'] = df_plot['Category'].apply(lambda x : x.split(',')[0])

	# Create a bar chart using Plotly
	fig = px.bar(df_plot, x='Category', y='Value', title='Category Values', range_y=[0, 1500])
	fig.show()


def adjust_tag(df, tag_name, min_n, max_n, verbose):
	# find the top 1026 samples where "Action-platformer" is at the top	
	list1 = df[['_id', '_tags']].values.tolist()
	total = list()
	for x in list1:
		# i = x[0]
		i = [j[0] for j in list1].index(x[0])
		index_df = list1[i][0]
		i_val = [i_ for i_ in range(len(x[1])) if list1[i][1][i_][2]==tag_name]
		if i_val != []:
			total.append([i_val[0], index_df])
	# we need to change downgrade the target_tag in the following samples
	todo_list = sorted([[x[0], x[1]] for x in total])[min_n:max_n]
	if verbose : print('#', min_n, max_n, len(todo_list))
	todo_indexes = [x[1] for x in todo_list]
	df.loc[todo_indexes, '_tags'] = df.loc[todo_indexes, '_tags'].apply(lambda x : downgrade_tag(x, tag_name))
	return df


def adjust_distribution(df, max_tag_density, max_iter, store_list=False, verbose=False):
	df = df.copy()
	iter = 0
	counter = [x for x in count_df(df) if x[0]>(len(df)*max_tag_density)]
	df_list = list()
	# we generate the level_0 column
	# df = df.reset_index() 
	old_count = count_df(df)
	while len(counter) > 0:
		if verbose : print(iter, counter[0][1], counter)
		if store_list:
			# we store all the history for analytics
			df_list.append(df.copy())
		else:
			# we keep storing the last iteration
			df_list = df
		df = adjust_tag(df, counter[0][1], min_n=int(len(df)*max_tag_density), max_n=counter[0][0], verbose=verbose)
		# update count
		new_count = count_df(df)
		if old_count == new_count:
			break
		else:
			old_count = new_count
		counter = [x for x in count_df(df) if x[0]>(len(df)*max_tag_density)]
		iter += 1
		if iter > max_iter:
			if verbose : print('max iteration break')
			break
	# we remove level_0
	# df.pop('level_0') 
	return df_list