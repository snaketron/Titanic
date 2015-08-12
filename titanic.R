train <- read.csv(file = "/home/simo/Downloads/titanic/train.csv")
test <- read.csv(file = "/home/simo/Downloads/titanic/test.csv")

train.names <- read.csv(file = "/home/simo/Downloads/titanic/train.csv")
test.names <- read.csv(file = "/home/simo/Downloads/titanic/test.csv")

################# DELETE BAD COLS #################
require(randomForest)

train$Survived <- as.factor(train$Survived)
test$Survived <- NA
test$Survived <- as.factor(test$Survived)

################# NEW CABIN #################
# summary(lm(Age~., data = train))


################# NAME -> CABIN #################
# index <- which(train$SibSp != 0 | train$Parch != 0)
# names <- train$Name[index]
# 
# last.names <- c()
# for(n in names) {
#   last.names <- c(last.names, unlist(strsplit(as.character(n), split = ","))[1])
# }
# 
# train$last.names <- NA
# train$last.names[index] <- last.names
# 
# 
# 
# index <- which(test$SibSp != 0 | test$Parch != 0)
# names <- test$Name[index]
# 
# last.names <- c()
# for(n in names) {
#   last.names <- c(last.names, unlist(strsplit(as.character(n), split = ","))[1])
# }
# 
# test$last.names <- NA
# test$last.names[index] <- last.names




# require(plyr)
# last.names.freq <- plyr::count(last.names)
# common.last.names <- last.names.freq$x[which(last.names.freq$freq != 1)]
# 
# for(c in common.last.names) {
#   x <- which(train$last.names %in% as.character(c))
#   cabin.nr <- as.character(train$Cabin[x])
# }





##################### CABIN #################
# multi.cabin <- which(grepl(pattern = paste(" +", sep = ''), x = as.character(train$Cabin), ignore.case = T))
# 
# for(mc in multi.cabin) {
#   mc <- multi.cabin[4]
#   
#   cabin.names <- unlist(strsplit(as.character(train$Cabin[mc]), split = " "))
#   
#   last.name <- unlist(strsplit(as.character(train$Name[mc]), split = ","))[1]
#   
#   
#   which(train$last.names %in% last.name)
#   which(test$last.names %in% last.name)
#   
#   
#   train$Cabin[which(train$last.names %in% last.name)]
#   test$Cabin[which(test$last.names %in% last.name)]
# }


# for(ln in train$last.names) {
#   
# }
# train$last.names




##################### DISCRET CABIN #################
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


################# NAME -> AGE #################
train$Young <- F
train$Young[which(grepl(pattern = paste("+", "Master\\.", "+", sep = ''), x = as.character(train$Name), ignore.case = T) == T)] <- T
train$Young[which(grepl(pattern = paste("+", "Miss\\.", "+", sep = ''), x = as.character(train$Name), ignore.case = T) == T)] <- T

test$Young <- F
test$Young[which(grepl(pattern = paste("+", "Master\\.", "+", sep = ''), x = as.character(test$Name), ignore.case = T) == T)] <- T
test$Young[which(grepl(pattern = paste("+", "Miss\\.", "+", sep = ''), x = as.character(test$Name), ignore.case = T) == T)] <- T



################# IMPUTE AGE #################
summary(lm(Age~., data = train))

temp <- rbind(train, test)
temp <- temp[, c("Age", "Pclass", "SibSp", "Young", "Sex")]

rf.imput <- randomForest(Age~., data = temp, ntree = 2000, na.action = na.omit)
to.imput <- temp[which(is.na(temp$Age)), ]
x <- predict(object = rf.imput, newdata = to.imput, type = "response")
temp$Age[which(is.na(temp$Age))] <- x


#updated train
train$Age <- temp$Age[1:891]
test$Age <- temp$Age[892:1309]

rm(x)
rm(to.imput)
rm(temp)
rm(rf.imput)



################ OPTIMIZE LEVELS ###############
levels(test$Sex) <- levels(train$Sex)
levels(test$cabin.binary) <- levels(train$cabin.binary)
levels(test$Embarked) <- levels(train$Embarked)
levels(test$CabinDiscrete) <- levels(train$CabinDiscrete)
levels(test$Survived) <- levels(train$Survived)

train$Young <- as.factor(train$Young)
test$Young <- as.factor(test$Young)
levels(test$Young) <- levels(train$Young)
levels(test$Cabin) <- levels(train$Cabin)

################# REMOVE bad predictors #################
train$Name <- NULL
test$Name <- NULL

train$Ticket <- NULL
test$Ticket <- NULL

train$PassengerId <- NULL
test$PassengerId <- NULL

# train$SibSp <- NULL
# test$SibSp <- NULL

# train$Parch <- NULL
# test$Parch <- NULL
# 
# train$Embarked <- NULL
# test$Embarked <- NULL



estimateGoodObs <- function(data, test, iters = 100) {
  values <- c()
  predicts <- c()
  votes <- c()
  for(i in 1:iters) {
    rf <- randomForest(Survived~. , data = train, na.action = na.omit, ntree = 1000)
    index <- which(as.numeric(as.character(rf$predicted)) - as.numeric(as.character(rf$y)) != 0)
    
    p <- predict(rf, newdata = test, type = "response")
    v <- predict(rf, newdata = test, type = "vote")[, 1]
    predicts <- rbind(predicts, as.numeric(as.character(p)))
    votes <- rbind(votes, v)
    
    row <- rep(x = 0, times = nrow(train))
    row[index] <- 1
    values <- rbind(values, row)
    cat(i, "\n")
  }
  
  result <- list(values = values, predicts = predicts, votes = votes)
  return (result)
}



estimateGoodTest <- function(good.data, test, iters = 100) {
  predicts <- c()
  votes <- c()
  for(i in 1:iters) {
    rf <- randomForest(Survived~. , data = good.data, na.action = na.omit, ntree = 1000)
    
    p <- predict(rf, newdata = test, type = "response")
    v <- predict(rf, newdata = test, type = "vote")[, 1]
    predicts <- rbind(predicts, as.numeric(as.character(p)))
    votes <- rbind(votes, v)
    cat(i, "\n")
  }
  
  result <- list(predicts = predicts, votes = votes)
  return (result)
}



################### P1 ###############
require("snow")
require("doParallel")
require("foreach")
require("doMC")
registerDoMC(cores = 8)
fbr.result <- (foreach(p = 1:8) %dopar% estimateGoodObs(data = train, test = test, iters = 100))

#parsing
fbr.values <- c()
for(i in 1:8) {
  fbr.values <- rbind(fbr.values, fbr.result[[i]]$values)
}

fbr.values <- colSums(fbr.values)/800
plot(fbr.values)



### learn test bad
bad.train <- train[which(fbr.values > 0.3), ]
good.train <- train[-which(fbr.values > 0.3), ]

rf.full <- randomForest(Survived~. , data = train[, -c(6,8)], na.action = na.omit, ntree = 1000, importance = T)
rf.full$importance
p.full <- predict(rf.full, newdata = test, type = "response")

rf.good <- randomForest(Survived~. , data = good.train, na.action = na.omit, ntree = 1000)
p.good <- predict(rf.good, newdata = test, type = "response")



################### P2 ###############
registerDoMC(cores = 8)
egt <- (foreach(p = 1:8) %dopar% estimateGoodTest(good.data = good.train, test = test, iters = 100))

#parsing
egt.predicts <- c() 
egt.votes <- c() 
for(i in 1:8) {
  egt.predicts <- rbind(egt.predicts, egt[[i]]$predicts)
  egt.votes <- rbind(egt.votes, egt[[i]]$votes)
}

egt.median <- c()
for(i in 1:ncol(egt.votes)) {
  egt.median <- c(egt.median, median(egt.votes[, i]))
}

plot(egt.median)


### build model from train bad
bad.test <- test[which(egt.median < 0.7 & egt.median > 0.3), ]

################### P3 ###############



rf.problematic <- randomForest(Survived~. , data = bad.train[, c(1, 3)], na.action = na.omit, ntree = 1000, importance = T)
p.problematic <- predict(rf.problematic, newdata = bad.test, type = "response")

test$Survived <- p.good
test$Survived[which(egt.median < 0.7 & egt.median > 0.3)] <- as.numeric(as.character(p.problematic))

as.numeric(as.character(p.problematic)) - as.numeric(as.character(x))


train.final <- train
test.final <- test


################# EXPORT  #################
train <- read.csv(file = "/home/simo/Downloads/titanic/train.csv")
test <- read.csv(file = "/home/simo/Downloads/titanic/test.csv")

export.train <- data.frame(PassengerId = train$PassengerId, Survived = train.final$Survived, row.names = NULL)
export.test <- data.frame(PassengerId = test$PassengerId, Survived = test.final$Survived, row.names = NULL)

write.table(export.train, file = "export.train.csv", quote = F, sep = ",", row.names = F, col.names = T)
write.table(export.test, file = "export.test.csv", quote = F, sep = ",", row.names = F, col.names = T)























################# PREDICT (post) #################
tree.response <- predict(object = rf, newdata = test, type = "response")
tree.vote <- predict(object = rf, newdata = test, type = "vote")
tree.prob <- predict(object = rf, newdata = test, type = "prob")


test$Survived <- tree.response

test$Survived[intersect(x = which(tree.vote[, 1] >= 0.4 & tree.vote[, 1] <= 0.6), y = which(test$Sex %in% "male"))] <- 1
test$Survived[intersect(x = which(tree.vote[, 1] >= 0.4 & tree.vote[, 1] <= 0.6), y = which(test$Sex %in% "female"))] <- 0


train.final <- train
test.final <- test



################# PREDICT (post) #################
male.response <- predict(object = rf.post.male, newdata = test.males, type = "response")
female.response <- predict(object = rf.post.female, newdata = test.females, type = "response")






impute.total <- randomForest::rfImpute(x = total.data[, -1], y = total.data$Survived)
colnames(impute.train)[1] <- "Survived"
randomForest::randomForest(Survived~., data = impute.train, na.action = na.omit, ntree = 2000)








############################ SVM #########################
require("e1071")

svm.train <- e1071::svm(Survived~., data = train, type = "C-classification", cross = 50)

plot(svm.train$accuracies)
length(which(as.numeric(as.character(svm.train$fitted))-as.numeric(as.character(train$Survived)) != 0))/nrow(train)

prediction <- predict(svm.train, test[,-8])

y1 <- as.numeric(as.character(p.full)) 
y2 <- as.numeric(as.character(prediction))
y2.svm <- c(y2[1:152], 1, y2[153:417])

which(y1 - y2.svm != 0)

export.svm <- data.frame(PassengerId = test.names$PassengerId, Survived = y2.svm, row.names = NULL)
write.table(export.svm, file = "svm.csv", quote = F, sep = ",", row.names = F, col.names = T)


########### T
tuned <- tune.svm(Survived~., data = train, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-2:2))
summary(tuned)

tunes <- c()
for(c in colnames(train)[-1]) {
  temp <- new.train[, -which(colnames(train) %in% c)]
  tuned <- tune.svm(Survived~., data = temp, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-1:1))
  
  tunes <- rbind(tunes, c(c, tuned$best.performance))
}


tuned$best.performance
svm.better <- e1071::svm(Survived~., data = train, type = "C-classification", cost = 10, gamma = 0.1)
length(which(as.numeric(as.character(svm.better$fitted))-as.numeric(as.character(train$Survived)) != 0))/nrow(train)

rows <- c()
for(i in 1:100) {
  svm <- e1071::svm(Survived~., data = train, type = "C-classification", cost = 10, gamma = 0.1)
  row <- rep(0, nrow(train))
  row[which(as.numeric(as.character(svm.better$fitted))-as.numeric(as.character(train$Survived)) != 0)] <- 1
  rows <- rbind(rows, row)
}
plot(colSums(rows))
bad <- train[which(colSums(rows) > 0), ]

svm.bad.killer <- e1071::svm(Survived~., data = bad[, -c(2, 8, 9, 7, 6, 5)], type = "C-classification", cost = 1e-04, gamma = 1e-06)
# summary(tune.svm(Survived~., data = bad[, -c(2, 8, 9, 7, 6, 5)], type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-4:4)))
length(which(as.numeric(as.character(svm.bad.killer$fitted))-as.numeric(as.character(bad$Survived)) != 0))/nrow(bad)





rf.full <- randomForest(Survived~. , data = train, na.action = na.omit, ntree = 2000, importance = T)
bad.rf <- train[which(as.numeric(as.character(rf.full$predicted)) - as.numeric(as.character(rf.full$y)) != 0), ]
rf.bad.killer <- randomForest(Survived~. , data = bad.rf[, -c(2, 8, 9, 7, 6, 5)], na.action = na.omit, ntree = 2000, importance = T)


prediction <- predict(svm.better, test[,-9], type = "response")
prediction.rf <- predict(rf.full, test[,-9], type = "vote")

x <- which(prediction.rf[, 1] < 0.6 & prediction.rf[, 1] > 0.4)
bad.test <- test[x, ]
prediction.bad <- predict(svm.bad.killer, bad.test[,-9], type = "response")


preds <- as.numeric(as.character(prediction))
preds[x] <- as.numeric(as.character(prediction.bad))
preds.export <- c(preds[1:152], 1, preds[153:417])


export.svm <- data.frame(PassengerId = test.names$PassengerId, Survived = preds.export, row.names = NULL)
write.table(export.svm, file = "svm.2.csv", quote = F, sep = ",", row.names = F, col.names = T)





temp <- train
train$Cabin <- as.character(train$Cabin)
train$Cabin[which(train$Cabin == "")] <- NA
train$Cabin <- as.factor(train$Cabin)

test$Cabin <- as.character(test$Cabin)
test$Cabin[which(test$Cabin == "")] <- NA
test$Cabin <- as.factor(test$Cabin)
# 
# train$Cabin <- as.character(train$Cabin)
# train$Cabin[which(is.na(train$Cabin))] <- ""
# train$Cabin <- as.factor(train$Cabin)


train <- rfImpute(x = train[, -1], train$Survived)
colnames(train)[1] <- "Survived"

full <- rbind(train, test)
randomForest(Cabin~., data = full, na.action = na.omit)
full <- rfImpute(x = full[, -1], full$)

test <- rfImpute(x = test[, -8], test$Age)
colnames(test)[1] <- "Survived"

colnames(new.train)[1] <- "Survived"


train <- temp.train
test <- temp.test





