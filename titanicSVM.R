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



############################ svm #########################

# tune svm params
tuned <- tune.svm(Survived~., data = train, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-2:2))
summary(tuned)


# tune predictors (effect of removing predictor)
tunes <- c()
for(c in colnames(train)[-1]) {
  temp <- new.train[, -which(colnames(train) %in% c)]
  tuned <- tune.svm(Survived~., data = temp, type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-1:1))
  
  tunes <- rbind(tunes, c(c, tuned$best.performance))
}


# learn
svm <- e1071::svm(Survived~., data = train, type = "C-classification", cost = 10, gamma = 0.1)
length(which(as.numeric(as.character(svm$fitted))-as.numeric(as.character(train$Survived)) != 0))/nrow(train)


# get bad data


svm.bad <- e1071::svm(Survived~., data = bad[, -c(2, 8, 9, 7, 6, 5)], type = "C-classification", cost = 1e-04, gamma = 1e-06)
# summary(tune.svm(Survived~., data = bad[, -c(2, 8, 9, 7, 6, 5)], type = "C-classification", gamma = 10^(-6:-1), cost = 10^(-4:4)))
length(which(as.numeric(as.character(svm.bad$fitted))-as.numeric(as.character(bad$Survived)) != 0))/nrow(bad)


# export
export.svm <- data.frame(PassengerId = pass.id, Survived = preds.export, row.names = NULL)
write.table(export.svm, file = "svm.2.csv", quote = F, sep = ",", row.names = F, col.names = T)

