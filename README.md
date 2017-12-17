# kdd17_src
source code for preparing kdd17, updated 20170105
## Preprocessor
#### preprocessing.py
- Functions for simple processing of data, like preparing data for run LDA.
- Remove cold-start users, like those whose number of ratings is less than 5;
- Other funcs like get user average ratings, generate review-aspect-weights triplets.

#### yelp_extractor.py
- Extract data from raw data provide by the yelp dataset challenge.

#### lda.py
- LDA implementation based on gensim.

#### aspects\_extractor.py
- Extract aspects from reviews, which are prepared for building hin.

#### filter\_processor.py
- provide functions that generate all kinds of filtering results, like positive or negative bids, or aids.

#### samples\_processor.py
-scripts used to generate data for sample users
-currently used to generate cat, state, city, stars of bids


## HIN modules
#### hin\_db\_generator.py
- Provide functions and SQLs to create or update related tables in database, etc create tables, add indexes.
- Preseve all the entities and relations in sqlite3.

#### hin_operator.py
- Preserve a whole HIN in memory.
- APIs provided for conviniently obtaining entities or relations from database of hin.

#### data\_split.py
- split uids into blocks for parallel computing

## HIN modules
#### meta\_stru\_sim\_computation.py
- The core module that computes meta-structure-based similarity according to Algorithm 1 in KDD16 paper. 
- decreprecated because of the efficiency problem.

## Similarity and Feature Generation

#### mf.py
- The standard matrix factorization model.

#### mf\_features\_generator.py
- The script that call the mf.py to generate latent features of users and items form the meta-graph based similarities.

#### uub\_commu\_mat\_computation.py
- Provide functions that calculate U-*-U-B style meta-graph commuting matrix. i.e. only calculate the number of instances of meta-graph.

#### ubb\_commu\_mat\_computation.py
- Provide functions that calculate U-B-*-B style meta-graph commuting matrix. i.e. only calculate the number of instances of meta-graph.

#### ui\_sim\_computation\_ubb.py
- Calculate the PathSim based user-item similarity for U-B-*-B style meta-graph. Refer to the paper for the formula.
- B-*-B style commuting matrix res is calculated in-place.

#### bb\_sim\_computation.py
- Calculate the B-*-B style meta-path based commuting matrix res.
- Discarded because of the too many entries in saving.

#### cal\_commuting\_mat.py
- Provide functions that calculate different communting matrix for given meta-graph.
 
#### commu\_mat\_sim\_computation.py
- calculate similarity based on commuting matrix operation.
- Mainly used for Yelp sample data.

## Prediction Model
#### fm\_one\_path.py
- FM model to compute only one-path based similarities.

#### fm\_with\_fnorm.py
- FM model with frobenius norm

#### fm\_with\_glasso.py
- FM model with group lasso.
- two optimizing methods are provided: proximal gradient and accelerated proximal gradient.

## Utils
utils used.

#### utils.py
- reverse map
- generate sparse matrix given data, rows, cols

#### db_util.py
- APIs for excuting sqls in sqlite3.

#### str_util.py
- Providing functions to processing strings, e.g. unicode2str, str2unicode.

#### logging_util.py
- util to write log.
