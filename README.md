README for lipy-als
===================

INTRODUCTION
============
Alternative least square algorithm is one of the most popular method to solve matrix factorization and related recommendation
system/collaborative filtering problem. At LinkedIn, we use matrix factorization to compute member embeddings and job
embeddings based on member/job interactions, such as views and applies. Then with these embeddings, we can do down stream
analysis, such as similar members, similar jobs, and recommending jobs to members using jymbii-glmix.

Currently, [Spark ALS](http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html) is the off the shelf solution
for solving matrix factorization at industry scale data size. However, it only provides basic functionalities for this
problem like many other popular open source packages, while at LinkedIn, we need more customized functions based on ALS.
Specifically, when embeddings were computed for a given set of old members and jobs, there is no easy way to compute
embeddings for new members given their job interactions. Currently, Spark ALS have to recompute the full matrix
factorization jointly with both old members and new members. However, we need to compute the embeddings for new members
online given the embeddings of old members and jobs. This feature is tightly related to the cold start issue.

Ideally, it would be optimal to extend the original Spark ALS with this new feature. However, given the time constraint
for intern projects, I (yli3) implemented this multiproduct in python as a single machine version of ALS with this new
feature added. Besides the new feature, I also included the code for doing clustering with embeddings that guarantees
balanced cluster sizes, which is related to later glmix training and load balancing. This multiproduct is mostly served
as referencing purpose for my intern project, and is not thoroughly tested for large-scale production data.


DEPENDENCIES
============
* [Numpy](http://www.numpy.org/)(>=1.11.1).


USAGE
=====
The main running program is lipy-als_trunk/lipy-als/src/als.py . It has four subcommands:

* `train_als` which trains an ALS model given member/job interaction data.
* `new_member_embeddings` which computes the embeddings for new members given embedding of old members/jobs and the
member/job interaction data of new members. This is the main difference between Spark ALS and this multiproduct.
* `train_clustering` which trains a clustering model based on member or job embeddings. The clustering algorithm
guarantees the number of members or jobs within a cluster has upper bound and lower bound.
* `predict_clustering` which predicts the cluster ids of new embeddings given a trained clustering model.


train_als
---------
Example:
```
$ als.py train_als input_records.csv output_member_embeddings.csv output_job_embeddings.csv --rank 50 --num_iterations 10 --Lambda 0.01
```
**input_records.csv** The input file of member job interactions data in CSV format as (memberId, jobId, score). Example
files are given here [view1Apply1Day1_train.csv](https://iwww.corp.linkedin.com/wiki/cf/download/attachments/161753063/view1Apply1Day1_train.csv)
, which is the training part (e.g. old) of member job interaction data of one day at LinkedIn. This data assigns score 1
for views and applies. Typically 1-10 would be good for view or apply scores. It seems better for view score and apply
score to be the same.

**output_member_embeddings.csv** The output file of member embeddings in CSV format as (memberId, (memberEmbeddings)). The
first field is memberId, and all subsequent fields are numerical embeddings.

**output_job_embeddings.csv** The output file of job embeddings in CSV format as (jobId, (jobEmbeddings)). The
first field is jobId, and all subsequent fields are numerical embeddings.

**--rank** The latent dimension of embeddings. Optional, default is 50. Typically 50-100 would be good for 1 months of data.

**--num_iterations** The number of iterations for the ALS algorithm. Optional, default is 10. At least 10, and 20 is usually enough
to converge.

**--Lambda** Strength of regularization. Optional, default is 0.01. 0.01 is recommended from Spark tutorial.


new_member_embeddings
---------------------
Example:
```
$ als.py new_member_embeddings input_member_embeddings.csv input_job_embeddings.csv input_records.csv output_member_embeddings
```
**input_member_embeddings.csv** The input file of old member embeddings in CSV format as (memberId, (memberEmbeddings)).
This file can be obtained by running train_als on old member job interaction data. The first field is memberId, and all
subsequent fields are numerical embeddings.

**input_job_embeddings.csv** The input file of old job embeddings in CSV format as (jobId, (jobEmbeddings)). This file
can be obtained by running train_als on old member job interaction data.The first field is jobId, and all subsequent
fields are numerical embeddings.

**input_records.csv** The input file of new member job interactions data in CSV format as (memberId, jobId, score). Example
files are given here [view1Apply1Day1_test.csv](https://iwww.corp.linkedin.com/wiki/cf/download/attachments/161753063/view1Apply1Day1_test.csv)
, which is the testing part (e.g. new) of member job interaction data of one day at LinkedIn. This data assigns score 1
for views and applies. Typically 1-10 would be good for view or apply scores. It seems better for view score and apply
score to be the same.

**output_member_embeddings.csv** The output file of new member embeddings in CSV format as (memberId, (memberEmbeddings)).
The first field is memberId, and all subsequent fields are numerical embeddings.


train_clustering
----------------
Example:
```
$ als.py train_clustering input_embeddings.csv output_cluster_ids.csv output_tree.pkl --num_max 10000 --num_min 1000
```
**input_embeddings.csv** The input file of member or job embeddings in CSV format as (embeddingId, (embeddings)). The
first field is embeddingId (memberId or jobId), and all subsequent fields are numerical embeddings.

**output_cluster_ids.csv** The output file of clustering results in CSV format as (embeddingId, clusterId). The
first field is embeddingId (memberId or jobId), and the second field is the associated clusterId.

**output_tree.pkl** The output file of clustering structure(tree) in python pickle format. This file is useful to predict
clusterId for new embeddings.

**--num_max** The maximum number of data points allowed within each cluster. Optional, default is 10000. --num_max is
recommended to be 10 times as --num_min

**--num_min** The minimum number of data points allowed within each cluster. Optional, default is 1000.


predict_clustering
----------------
Example:
```
$ als.py predict_clustering input_embeddings.csv input_tree.pkl output_cluster_ids.csv
```
**input_embeddings.csv** The input file of new member or job embeddings in CSV format as (embeddingId, (embeddings)). The
first field is embeddingId (memberId or jobId), and all subsequent fields are numerical embeddings.

**input_tree.pkl** The input file of clustering structure(tree) trained based on old member or job embeddings in python
pickle format. This file is generated by train_clustering.

**output_cluster_ids.csv** The output file of clustering results for new member or job embeddings in CSV format as
(embeddingId, clusterId). The first field is embeddingId (memberId or jobId), and the second field is the associated clusterId.



CASE STUDY
==========
Download [view1Apply1Day1_train.csv](https://iwww.corp.linkedin.com/wiki/cf/download/attachments/161753063/view1Apply1Day1_train.csv)
as old member job interaction data, and [view1Apply1Day1_test.csv](https://iwww.corp.linkedin.com/wiki/cf/download/attachments/161753063/view1Apply1Day1_test.csv)
as new member job interaction data.

1. **train_als**
```
$ python lipy-als_trunk/lipy-als/src/als.py train_als view1Apply1Day1_train.csv view1Apply1Day1_train_memberEmbeddings.csv view1Apply1Day1_train_jobEmbeddings.csv --rank 50 --num_iterations 10 --Lambda 0.01
```
This will train an ALS model using view1Apply1Day1_train.csv as input, and generate view1Apply1Day1_train_memberEmbeddings.csv
as member embeddings output and view1Apply1Day1_train_jobEmbeddings.csv as job embeddings output.

2. **new_member_embeddings**
```
$ python lipy-als_trunk/lipy-als/src/als.py new_member_embeddings view1Apply1Day1_train_memberEmbeddings.csv view1Apply1Day1_train_jobEmbeddings.csv view1Apply1Day1_test.csv view1Apply1Day1_test_memberEmbeddings.csv
```
This will load the old member embeddings view1Apply1Day1_train_memberEmbeddings.csv, the old job embeddings view1Apply1Day1_train_jobEmbeddings.csv,
and the new member job interactions view1Apply1Day1_test.csv as input, and generate new member embeddings view1Apply1Day1_test_memberEmbeddings
as output.

3. **train_clustering**
```
$ python lipy-als_trunk/lipy-als/src/als.py train_clustering view1Apply1Day1_train_memberEmbeddings.csv view1Apply1Day1_train_memberClusterIds.csv view1Apply1Day1_train_memberTree.pkl --num_max 10000 --num_min 1000
```
This will load the old member embeddings view1Apply1Day1_train_memberEmbeddings.csv as input, and generate
view1Apply1Day1_train_memberClusterIds.csv as the clustering results with each memberId associated with one clusterId,
and generate the view1Apply1Day1_train_memberTree.pkl as the clustering tree structure saved as python pickle format.
This clustering guarantees that each cluster has at most 10000 members and at least 1000 members.

4. **predict_clustering**
```
$ python lipy-als_trunk/lipy-als/src/als.py predict_clustering view1Apply1Day1_test_memberEmbeddings.csv view1Apply1Day1_train_memberTree.pkl view1Apply1Day1_test_memberClusterIds.csv
```
This will load new member embeddings view1Apply1Day1_test_memberEmbeddings.csv and the previously generated clustering
tree structure view1Apply1Day1_train_memberTree.pkl as input, and generate the view1Apply1Day1_test_memberClusterIds.csv
as the clustering results for new members with each memberId associated with one clusterId.
