library(klaR)
library(e1071)
library(tm)
library(MASS)
library(caret)
library(SnowballC)

train = read.table("train_set.csv", sep="\t", header=T)
test = read.table("test_set.csv", sep="\t", header=T)

#pred <- train$pred
#hw <- subset(train, train$class==1)
#sw <- subset(train, train$class==0)

#train$set = "Hardware"
#train[train$class==0,]$set = "Software"
#train$set = as.factor(train$set)

#nb = NaiveBayes(train[ ,train$pred], train$set)

#stopwords
mystopwords <- c(stopwords("english"),"week","arduino","words","need","get","will","want","know","work","also")

#corpus for train set
train.corpus <- Corpus(VectorSource(train$pred))
train.corpus <- tm_map(train.corpus, content_transformer(tolower))
train.corpus <- tm_map(train.corpus, removePunctuation)
train.corpus <- tm_map(train.corpus, stripWhitespace)
train.corpus <- tm_map(train.corpus, removeNumbers)
train.corpus <- tm_map(train.corpus, removeWords, mystopwords)
train.corpus <- tm_map(train.corpus, stemDocument)
train.corpus <- tm_map(train.corpus, removeWords, "(http)\\w+")
train.corpus <- tm_map(train.corpus, removeWords, "\\b[a-zA-Z0-9]{10,100}\\b")
#train.corpus <- gsub('(http)\\w+',"",train.corpus)
train.corpus.dtm <- DocumentTermMatrix(train.corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE, removePunctuation=TRUE))
train.corpus.dtms <- removeSparseTerms(train.corpus.dtm, 0.935)

#TermDocumentMatrix(train.corpus)
#inspect(train.corpus.dtm)
#findFreqTerms(train.corpus.dtm, N)   #N <- freq

#corpus for test set
test.corpus <- Corpus(VectorSource(test$pred))
test.corpus <- tm_map(test.corpus, content_transformer(tolower))
test.corpus <- tm_map(test.corpus, removePunctuation)
test.corpus <- tm_map(test.corpus, stripWhitespace)
test.corpus <- tm_map(test.corpus, removeNumbers)
test.corpus <- tm_map(test.corpus, removeWords, mystopwords)
test.corpus <- tm_map(test.corpus, stemDocument)
test.corpus <- tm_map(test.corpus, removeWords, "(http)\\w+")
test.corpus <- tm_map(test.corpus, removeWords, "\\b[a-zA-Z0-9]{10,100}\\b")
test.corpus.dtm <- DocumentTermMatrix(test.corpus, control = list(weighting = function(x) weightTfIdf(x, normalize = FALSE), stopwords = TRUE, removePunctuation=TRUE))
test.corpus.dtms <- removeSparseTerms(test.corpus.dtm, 0.935) 


m <- as.matrix(train.corpus.dtms)
n <- as.matrix(test.corpus.dtms)
#v <- sort(rowSums(m), decreasing=TRUE)

#model <- naiveBayes(set ~ pred, data = train)
#pred <- predict(model, train[495:510,])
#pred1 <- predict(model, newdata = data.frame(pred="This post is related to Software"), type="raw")
#pred2 <- predict(model, newdata = data.frame(pred="Hardware"), type="raw")
#pred3 <- predict(model, newdata = test[1,], type="raw")
#predict(model, newdata = data.frame(Sepal.Width = 3), type=('raw'))
#table(pred, test)

#Train model
#tune.control <- tune.control(random =F, nrepeat=1, repeat.aggregate=min,sampling=c("cross"),sampling.aggregate=mean, cross=10, best.model=T, performances=T) 
#model <- naiveBayes(m,as.factor(train$class),tune.control);

#Prediction

#"my.predict.NB"<-function(object,newdata){ 
#results <- predict(model,n[10:20,])
#} 
#test <- chisq.test(as.factor(train$class)[496:505], results)
#print(test)
#tune(naiveBayes,m,as.factor(train$class),fun=my.predict.NB)

#mat <- table(results,n[10:20])
#write.table(mat, file="confusion_matrix.txt", row.names=FALSE, col.names=FALSE)
#write.matrix(mat,'confusion_matrix.txt',sep = "\t")

fitcontrol <- trainControl(method="cv", 10)
#set.seed(825)
nb <- train(m, as.factor(train$class), method="nb", tuneGrid=data.frame(.fL=1, .usekernel=FALSE), trControl=fitcontrol)

print(nb$results)

print(nb$resample)
