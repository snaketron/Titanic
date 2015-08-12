##################### inputs and dependencies #################
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")

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




################# remove bad cols #################
train$Name <- NULL
test$Name <- NULL

train$Ticket <- NULL
test$Ticket <- NULL

train$PassengerId <- NULL
test$PassengerId <- NULL



################### initial learning ###############
rf.full <- randomForest(Survived~. , data = train, na.action = na.omit, ntree = 1000, importance = T)
predict.full <- predict(rf.full, newdata = test, type = "response")



################### get outlying/good observations (both train/test data) ###############

analyzeObservations <- function(data, test, iters = 100) {
  values <- c()
  predicts <- c()
  votes <- c()
  for(i in 1:iters) {
    rf <- randomForest(Survived~. , data = train, na.action = na.omit, ntree = 1000)
    index <- which(as.numeric(as.character(rf$predicted)) - as.numeric(as.character(rf$y)) != 0)
    
    v <- predict(rf, newdata = test, type = "vote")[, 1]
    votes <- rbind(votes, v)  
    row <- rep(x = 0, times = nrow(train))
    row[index] <- 1
    values <- rbind(values, row)
    cat(i, "\n")
  }
  
  result <- list(values = values, votes = votes)
  return (result)
}


analyzeTestObservations <- function(train, test, iters = 100) {
  predicts <- c()
  votes <- c()
  for(i in 1:iters) {
    rf <- randomForest(Survived~. , data = train, na.action = na.omit, ntree = 1000)
    
    p <- predict(rf, newdata = test, type = "response")
    v <- predict(rf, newdata = test, type = "vote")[, 1]
    predicts <- rbind(predicts, as.numeric(as.character(p)))
    votes <- rbind(votes, v)
    cat(i, "\n")
  }
  
  result <- list(predicts = predicts, votes = votes)
  return (result)
}



registerDoMC(cores = 8)
observations <- (foreach(p = 1:8) %dopar% analyzeObservations(data = train, test = test, iters = 100))


# get wrong predictions for train data
wrong.predictions <- c()
for(i in 1:8) {
  wrong.predictions <- rbind(wrong.predictions, observations[[i]]$values)
}
wrong.predictions <- colSums(wrong.predictions)/800
plot(wrong.predictions)


# get votes
votes.predictions <- colMeans(observations$votes)


# good and bad train observations
bad.train <- train[which(votes.predictions > 0.3), ]
good.train <- train[-which(votes.predictions > 0.3), ]


# good/bad learning
rf.good <- randomForest(Survived~. , data = good.train, na.action = na.omit, ntree = 1000)
rf.bad <- randomForest(Survived~. , data = bad.train, na.action = na.omit, ntree = 1000)


# get wrong predictions for test data
registerDoMC(cores = 8)
test.observations <- (foreach(p = 1:8) %dopar% analyzeTestObservations(train = train, test = test, iters = 100))


#parsing
test.predictions <- c() 
test.votes <- c() 
for(i in 1:8) {
  test.predictions <- rbind(test.predictions, test.observations[[i]]$predicts)
  test.votes <- rbind(test.votes, test.observations[[i]]$votes)
}

test.predictions <- colSums(test.predictions)
test.votes <- colMeans(test.votes)


# build model from train bad
bad.test <- test[which(test.votes < 0.7 & test.votes > 0.3), ]


# predict bad observations
rf.bad <- randomForest(Survived~. , data = bad.train, na.action = na.omit, ntree = 1000, importance = T)
predict.bad <- predict(rf.bad, newdata = bad.test, type = "response")


test$Survived <- predict.full
test$Survived[which(test.votes < 0.7 & test.votes > 0.3)] <- as.numeric(as.character(predict.bad))



###################### export results ############################## 

export.test <- data.frame(PassengerId = pass.id, Survived = test$Survived, row.names = NULL)
write.table(export.test, file = "rf.csv", quote = F, sep = ",", row.names = F, col.names = T)
