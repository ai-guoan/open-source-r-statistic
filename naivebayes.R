#########################################################################
#   File Name       : naivebayes.py
#########################################################################
#   Project/Product : All
#   Title           : sms classification
#   Author          : Guoan.Li
#########################################################################
#   Description     : sms classificate by naive bayes
#                     load library:e1071,tm,SnowballC,wordcloud,gmodels
#
#########################################################################
#   Revision History:
# 
#   Version     Date          Initials      CR#          Descriptions
#   ---------   ----------    ------------  ----------   ---------------
#   1.0         29/03/2018    Guoan.Li      N/A          Original
#########################################################################

# prepare dataset
setwd('d:\\data')
sms_raw <- read.csv("D:\\data\\sms_spam.csv", stringsAsFactors = FALSE,encoding='UTF-8')
names(sms_raw)[1]<-'type'
sms_raw$type <- factor(sms_raw$type)

library(tm)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
lapply(sms_corpus[1:3], as.character)

sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower)) # small letters
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers) # remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords()) # remove stop words
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation) # remove punctuation

removePunctuation("hello...world")
replacePunctuation <- function(x) { gsub("[[:punct:]]+", " ", x) }
replacePunctuation("hello...world")

library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))

sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument) # forms
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace) # eliminate unneeded whitespace

lapply(sms_corpus[1:3], as.character)
lapply(sms_corpus_clean[1:3], as.character)

sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm_train <- sms_dtm[1:4169, ]          #train set
sms_dtm_test  <- sms_dtm[4170:5559, ]       #test set

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels  <- sms_raw[4170:5559, ]$type

library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, random.order = FALSE)

spam <- subset(sms_raw, type == "spam")
ham  <- subset(sms_raw, type == "ham")

wordcloud(spam$text, max.words = 50, scale = c(3, 0.5))
wordcloud(ham$text, max.words = 50, scale = c(3, 0.5))

sms_dtm_freq_train <- removeSparseTerms(sms_dtm_train, 0.999) # remove 99.9% 

sms_freq_words <- findFreqTerms(sms_dtm_train, 5) # save high frequency words

sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
} # yes or no

sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test  <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

# naive bayes classification
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)

# performance
sms_test_pred <- predict(sms_classifier, sms_test)

library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))

# improve
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_test_pred2 <- predict(sms_classifier2, sms_test)
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))
