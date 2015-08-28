##################### inputs and dependencies #################
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")

require("e1071")
require("randomForest")
require("snow")
require("doParallel")
require("foreach")
require("doMC")

train$Survived <- as.factor(train$Survived)
test$Survived <- NA
test$Survived <- as.factor(test$Survived)

pass.id <- test$PassengerId




##################### cluster cabins #################

train$Cabin <- as.character(train$Cabin)
decks <- LETTERS
for(d in decks) {
  train$Cabin[which(grepl(pattern = paste(d, "+", sep = ''), x = as.character(train$Cabin), ignore.case = T) == T)] <- d
}
train$Cabin <- as.factor(train$Cabin)


test$Cabin <- as.character(test$Cabin)
decks <- LETTERS
for(d in decks) {
  test$Cabin[which(grepl(pattern = paste(d, "+", sep = ''), x = as.character(test$Cabin), ignore.case = T) == T)] <- d
}
test$Cabin <- as.factor(test$Cabin)

rm(d)
rm(decks)




################# impute age #################
train$Young <- F
train$Young[which(grepl(pattern = paste("+", "Master\\.", "+", sep = ''), x = as.character(train$Name), ignore.case = T) == T)] <- T
train$Young[which(grepl(pattern = paste("+", "Miss\\.", "+", sep = ''), x = as.character(train$Name), ignore.case = T) == T)] <- T

test$Young <- F
test$Young[which(grepl(pattern = paste("+", "Master\\.", "+", sep = ''), x = as.character(test$Name), ignore.case = T) == T)] <- T
test$Young[which(grepl(pattern = paste("+", "Miss\\.", "+", sep = ''), x = as.character(test$Name), ignore.case = T) == T)] <- T

getTitle <- function(data) {
  sp <- strsplit(as.character(data$Name), split = "\\.")
  sp <- unlist(sp)
  
  r <- c()
  for(s in sp) {
    r <- c(r, unlist(strsplit(x = s, split = "\\, "))[2])
  }
  r <- r[!is.na(r)]
  plyr::count(r)
  return (r)
}
train$Title <- as.factor(getTitle(data = train))
test$Title <-  as.factor(getTitle(data = test))


train$Alone <- ifelse(test = train$SibSp == 0 & train$Parch == 0, yes = T, no = F)
test$Alone <- ifelse(test = test$SibSp == 0 & test$Parch == 0, yes = T, no = F)


summary(lm(Age~Title, data = train))
randomForest(Survived~Title+Sex+Pclass+Age+Young+SibSp+Parch+Alone, data = train, ntree = 2000, importance = T)

temp <- rbind(train, test)
temp <- temp[, c("Age", "Pclass", "SibSp", "Young", "Sex", "Title", "Alone")]

age.predicts <- c()
for(i in 1:100) {
  train.temp <- temp[!is.na(temp$Age), ]
  test.temp <- temp[is.na(temp$Age), ]
  
  rf.imput <- randomForest(y = train.temp$Age, x = train.temp[, -1], ntree = 2000, na.action = na.omit)
  x <- predict(object = rf.imput, newdata = test.temp[, -1], type = "response")
  age.predicts <- rbind(age.predicts, x)
  
  cat("i:", i, "\n")
  
}
temp$Age[which(is.na(temp$Age))] <- apply(age.predicts, MARGIN = 2, FUN = median)



#updated train
train$Age <- temp$Age[1:891]
test$Age <- temp$Age[892:1309]

rm(x)
rm(temp)
rm(rf.imput)
rm(test.temp)
rm(train.temp)
rm(age.predicts)
rm(i)



################ relevel ###############
levels(test$Sex) <- levels(train$Sex)
levels(test$cabin.binary) <- levels(train$cabin.binary)
levels(test$Embarked) <- levels(train$Embarked)
levels(test$CabinDiscrete) <- levels(train$CabinDiscrete)
levels(test$Survived) <- levels(train$Survived)

train$Young <- as.factor(train$Young)
test$Young <- as.factor(test$Young)
levels(test$Young) <- levels(train$Young)
levels(test$Cabin) <- levels(train$Cabin)
levels(test$Title) <- levels(train$Title)
levels(test$Alone) <- levels(train$Alone)


################# remove bad cols #################
train$Name <- NULL
test$Name <- NULL

train$Ticket <- NULL
test$Ticket <- NULL

train$PassengerId <- NULL
test$PassengerId <- NULL



############################ svm #########################

# tune svm params
tuned <- tune.svm(Survived~Title+Sex+Pclass+Age+Young+SibSp+Parch+Cabin+Alone, data = train, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-2:2))
summary(tuned)
rm(tuned)


# tune predictors (effect of removing predictor)
# tunes <- c()
# for(c in colnames(train)[-1]) {
#   temp <- new.train[, -which(colnames(train) %in% c)]
#   tuned <- tune.svm(Survived~., data = temp, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-1:1))
#   
#   tunes <- rbind(tunes, c(c, tuned$best.performance))
# }


# learn
svm <- e1071::svm(Survived~Title+Sex+Pclass+Age+Young+SibSp+Parch+Cabin, data = train, type = "C-classification", cost = 1, gamma = 0.1, cross = 500)
predict <- predict(object = svm, newdata = train, type = "response")
table(predict, train$Survived)
rm(predict)



compositeTraining <- function() {
  
  crossValidationLearning <- function(data, boots, ratio) {
    indices <- 1:nrow(data)
    sample.size <- round(x = length(indices)*ratio, digits = 0)
    result.svm <- c()
    result.rf <- c()
    
    for(b in 1:boots) {
      sampled.indices <- sample(x = indices, size = sample.size, replace = F)
      temp <- data[sampled.indices, ]
      
      svm <- e1071::svm(Survived~., data = temp, type = "C-classification", cost = 1, gamma = 0.1)
      rf <- randomForest(Survived~., data = temp, ntree = 2000)
      bad.observations.svm <- which(as.numeric(as.character(svm$fitted))-as.numeric(as.character(temp$Survived)) != 0)
      bad.observations.rf <- which(as.numeric(as.character(rf$predicted))-as.numeric(as.character(temp$Survived)) != 0)
      result.svm <- c(result.svm, bad.observations.svm)
      result.rf <- c(result.rf, bad.observations.rf)
      
      cat(b, "\n")
    }
    
    result <- list(result.rf = result.rf, result.svm = result.svm)
    return (result)
  }
  
  
  cvl <- crossValidationLearning(data = train, boots = 100, ratio = 0.5)
  cvl.freq.rf <- plyr::count(cvl$result.rf)
  cvl.freq.rf <- cvl.freq.rf[order(cvl.freq.rf$freq, decreasing = T), ]
  
  cvl.freq.svm <- plyr::count(cvl$result.svm)
  cvl.freq.svm <- cvl.freq.svm[order(cvl.freq.svm$freq, decreasing = T), ]
  rm(cvl)
  
  plot(cvl.freq.rf$freq)
  plot(cvl.freq.svm$freq)
  intersect(cvl.freq.rf$x, cvl.freq.svm$x)
  
  
  bad <- train[cvl.freq.svm$x, ]
  bad <- train[cvl.freq.svm$freq > 20, ]
  table(bad[, c(1,3)])
  table(train[, c(1,3)])
  
  
  
  
  
  
  temp.train <- train
  temp.test <- test
  
  rf <- randomForest(Survived~., data = train, importance = T, ntree = 2000)
  rf$importance
  
  svm <- e1071::svm(Survived~., data = train, type = "C-classification", cost = 1, gamma = 0.1)
  predict <- predict(object = svm, newdata = train, type = "response")
  table(predict, train$Survived)
  rm(predict)
  
  
  train$Embarked <- NULL
  test$Embarked <- NULL
  
  train$Parch <- NULL
  test$Parch <- NULL
  
  
  
#   train$Fare <- NULL
#   test$Fare <- NULL
#   
#   train$Young <- NULL
#   test$Young <- NULL
#   
#   train$SibSp <- NULL
#   test$SibSp <- NULL
  
  cvl <- crossValidationLearning(data = train, boots = 1000, ratio = 0.5)
  cvl.freq <- plyr::count(cvl)
  cvl.freq <- cvl.freq[order(cvl.freq$freq, decreasing = T), ]
  rm(cvl)
  
  
  
  train.bad <- train[cvl.freq$x, ]
  train.bad$Pclass <- NULL
  
  
  rf <- randomForest(Survived~., data = train.bad, importance = T, ntree = 2000)
  svm <- e1071::svm(Survived~., data = train.bad, type = "C-classification", cost = 1, gamma = 0.1)
  cvl <- crossValidationLearning(data = train.bad, boots = 1000, ratio = 0.5)
  cvl.freq <- plyr::count(cvl)
  cvl.freq <- cvl.freq[order(cvl.freq$freq, decreasing = T), ]
  rm(cvl)
  
  
  predict <- predict(object = svm, newdata = train.bad, type = "response")
  table(predict, train.bad$Survived)
  rm(predict)
  
  train.bad <- train[cvl.freq$x, ]
  table(train.bad[, c(1, 3)])
  cvl <- crossValidationLearning(data = train.bad, boots = 1000, ratio = 0.5)
  cvl.freq <- plyr::count(cvl)
  cvl.freq <- cvl.freq[order(cvl.freq$freq, decreasing = T), ]
  rm(cvl)
  
  
  
}



svm <- e1071::svm(Survived~Title+Sex+Pclass+Age+Young+SibSp+Parch+Alone, data = train, type = "C-classification", cost = 10, gamma = 0.01)
predict <- predict(object = svm, newdata = train, type = "response")
table(predict, train$Survived)
rm(predict)


which(as.numeric(as.character(svm$fitted))-as.numeric(as.character(train$Survived)) != 0)
bad <- train[which(as.numeric(as.character(svm$fitted))-as.numeric(as.character(train$Survived)) != 0), ]

# get bad data
predict.svm <- predict(object = svm, newdata = test[, -9], type = "response")
predict.svm <- as.numeric(as.character(predict.svm))

# export
export.svm <- data.frame(PassengerId = pass.id, Survived = predict.svm, row.names = NULL)
write.table(export.svm, file = "svm.csv", quote = F, sep = ",", row.names = F, col.names = T)





rf <- randomForest(Survived~Title+Sex+Pclass+Age+Young+SibSp+Parch+Alone, data = train, ntree = 2000, importance = T)
predict.rf <- predict(object = rf, newdata = test[, -9], type = "response")
predict.rf <- as.numeric(as.character(predict.rf))


export.rf <- data.frame(PassengerId = pass.id, Survived = predict.rf, row.names = NULL)
write.table(export.rf, file = "rf.csv", quote = F, sep = ",", row.names = F, col.names = T)




which(predict.rf != predict.svm)

which(predict.svm != predict.rf)

intersect(predict.rf, predict.svm)

rf.prob <- predict(object = rf, newdata = test[, -9], type = "prob")
predict(object = svm, newdata = test[, -9], type = "prob")


which(predict.rf == predict.svm)
which(predict.rf == predict.svm)

predict.rf[which(rf.prob[, 1] < 0.2)] <- predict.svm[which(rf.prob[, 1] < 0.2)]

which(predict.rf == 1 & predict.svm == 0)
