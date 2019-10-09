from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, estimate_bandwidth, SpectralClustering, \
    AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import warnings

warnings.filterwarnings('ignore')

# newsgroups_train = fetch_20newsgroups(subset='train')
# # 总共有20个类
# print(newsgroups_train.keys())
# print(list(newsgroups_train.target_names))

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='all', categories=categories)
print(newsgroups_train.keys())
# print(list(newsgroups_train.data))
y = newsgroups_train.target
# TF: 词频 TF(w)=(词w在文档中出现的次数)/(文档的总次数)
# IDF: 逆向文件频率。有些词虽然频繁出现，但是信息量小，如is,of,that等单词，我们可以用IDF(w)=log_e(总文档数)/
# (词w出现的文档数+1)，这样可以降低频繁出现的那些词的权重
# TF-IDF就是将上述两个量相乘
# 这样就可以把文字转为向量
vectorizer = TfidfVectorizer()

# vectorizer = HashingVectorizer()

# vectorizer = make_pipeline(HashingVectorizer(),TfidfTransformer())
vectors = vectorizer.fit_transform(newsgroups_train.data)
svd = TruncatedSVD(n_components=5)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)
X = lsa.fit_transform(vectors)

estimator_1 = KMeans(n_clusters=4)
estimator_1.fit(X)
y_predict_1 = estimator_1.labels_

acc_1 = metrics.accuracy_score(y, y_predict_1)
NMI_1 = metrics.normalized_mutual_info_score(y, y_predict_1)
homogeneity_1 = metrics.homogeneity_score(y, y_predict_1)
completeness_1 = metrics.completeness_score(y, y_predict_1)

# 大概preference取这个值结果比较好，参考度越大，每个点作为中心的可能性越大，聚类的簇也越多，所以这里小点较好
estimator_2 = AffinityPropagation(preference=-30000)
estimator_2.fit(X)
y_predict_2 = estimator_2.labels_

acc_2 = metrics.accuracy_score(y, y_predict_2)
NMI_2 = metrics.normalized_mutual_info_score(y, y_predict_2)
homogeneity_2 = metrics.homogeneity_score(y, y_predict_2)
completeness_2 = metrics.completeness_score(y, y_predict_2)

estimator_3 = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
estimator_3.fit(X)
y_predict_3 = estimator_3.labels_

acc_3 = metrics.accuracy_score(y, y_predict_3)
NMI_3 = metrics.normalized_mutual_info_score(y, y_predict_3)
homogeneity_3 = metrics.homogeneity_score(y, y_predict_3)
completeness_3 = metrics.completeness_score(y, y_predict_3)

estimator_4 = AgglomerativeClustering(n_clusters=4, linkage='ward')
estimator_4.fit(X)
y_predict_4 = estimator_4.labels_

acc_4 = metrics.accuracy_score(y, y_predict_4)
NMI_4 = metrics.normalized_mutual_info_score(y, y_predict_4)
homogeneity_4 = metrics.homogeneity_score(y, y_predict_4)
completeness_4 = metrics.completeness_score(y, y_predict_4)

estimator_5 = DBSCAN(eps=0.5, min_samples=80)
estimator_5.fit(X)
y_predict_5 = estimator_5.labels_

acc_5 = metrics.accuracy_score(y, y_predict_5)
NMI_5 = metrics.normalized_mutual_info_score(y, y_predict_5)
homogeneity_5 = metrics.homogeneity_score(y, y_predict_5)
completeness_5 = metrics.completeness_score(y, y_predict_5)
#
# y_predict = estimator.labels_
gmmModel = GaussianMixture(n_components=4, covariance_type='diag', random_state=0)
gmmModel.fit(X)
y_predict = gmmModel.predict(X)
# print(y_predict)
# print(max(y_predict))

acc = metrics.accuracy_score(y, y_predict)
NMI = metrics.normalized_mutual_info_score(y, y_predict)
homogeneity = metrics.homogeneity_score(y, y_predict)
completeness = metrics.completeness_score(y, y_predict)

bandwidth = estimate_bandwidth(X, quantile=0.9, n_samples=1000)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
y_predict_6 = ms.labels_

acc_6 = metrics.accuracy_score(y, y_predict_6)
NMI_6 = metrics.normalized_mutual_info_score(y, y_predict_6)
homogeneity_6 = metrics.homogeneity_score(y, y_predict_6)
completeness_6 = metrics.completeness_score(y, y_predict_6)

print('Acc:', acc_1, acc_2, acc_6, acc_3, acc_4, acc_5, acc)
print('NMI:', NMI_1, NMI_2, NMI_6, NMI_3, NMI_4, NMI_5, NMI)
print('Homogeneity:', homogeneity_1, homogeneity_2, homogeneity_6, homogeneity_3, homogeneity_4, homogeneity_5,
      homogeneity)
print('Completeness:', completeness_1, completeness_2, completeness_6, completeness_3, completeness_4, completeness_5,
      completeness)
