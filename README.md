# Python-Sklearn-Implementation-
Classifying Stack Exchange topics.

The project aims at classifying the various questions that are posted in websites such into its relevant subdomains, e.g arduino posts are classifed into domains of hardware or software. i.e. if a post is relevant to a hardware problem or a software problem.

Dataset used : data dump of askarduino.com (arduino.stackexchange.com)
This data dump consists of 7 files which are in .xml format. For our purpose posts.xml and tags.xml was used. In the posts.xml the postId, postTypeId (1 : question, 2 : answer), title of the post, body of the post, tags in the post , score, comments Count, view Count, answer Count, favorite Count, creation Date can be found for the posts. In the file tags.xml the tags and its counts can be found.

To build a training set for the text classifier it was necessary to have a labelled dataset of posts as a hardware post or a software post which were labelled as 1 and 0 respectively. We did the labelling of the posts by first manually labelling the tags that were present in tags.xml. There were a total of 203 tags in tags.xml and 98 of them were relevant to arduino hardware and the rest 105 were relevant to arduino software. After the manual labelling of the tags a program was made which labelled a post which had more hardware tags as a hardware post (1) or if more software tags as a software post (0). Out of 2262 posts that were taken from posts.xml 500 posts which were labelled ashardware (1) and 500 posts which were labelled as software (0) were taken for making the
training set of the classifier and the remaining 1262 posts were used for testing. Hence a train set of total 1000 posts was made which had 500 labelled as hardware and 500 labelled as software.

The training set contains posts from arduino.stackexchange.com labelled as hardware (1) or software (0). 3 models are used to classify them. Grid Search CV is used to find the optimal parameters for the models, such as Regularization Penalty in LinearSVM and Logreg, Smoothng in NaiveBayes. Principal Component Analysis (PCA) is used to Visualize the data after vectorization. Cross validation is used to evaluate the performance using train test split. Validation Curves are plotted to validate the performance results.
