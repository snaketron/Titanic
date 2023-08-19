##################### inputs and dependencies #################
train <- read.csv(file = "train.csv")
test <- read.csv(file = "test.csv")

require("e1071")
require("randomForest")
require("plyr")

train$Survived <- as.factor(train$Survived)
test$Survived <- NA
test$Survived <- as.factor(test$Survived)


p <- read.csv(file = "test.csv")
p$Survived <- NA
p$Survived <- as.factor(p$Survived)
pass.id <- p$PassengerId
rm(p)



##################### cluster cabins #################

train$Cabin <- as.character(train$Cabin)
decks <- LETTERS
for(d in decks) {
  train$Cabin[which(grepl(pattern = paste(d, "+", sep = ''), x = as.character(train$Cabin), ignore.case = T) == T)] <- d
}
train$Cabin <- as.factor(train$Cabin)
rm(d)
rm(decks)


test$Cabin <- as.character(test$Cabin)
decks <- LETTERS
for(d in decks) {
  test$Cabin[which(grepl(pattern = paste(d, "+", sep = ''), x = as.character(test$Cabin), ignore.case = T) == T)] <- d
}
test$Cabin <- as.factor(test$Cabin)
rm(d)
rm(decks)







################# get titles #################
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








############## get high class ###########
high.class <- c("the Countess", "Sir", "Mme", "Mlle", "Dr")
train$Title <- as.character(train$Title)
train$Title[which(train$Title %in% high.class)] <- "highclass"

test$Title <- as.character(test$Title)
test$Title[which(test$Title %in% high.class)] <- "highclass"

train$Title <- as.factor(train$Title)
test$Title <- as.factor(test$Title)
levels(test$Title) <- levels(train$Title)
rm(high.class)







########### get alone ####################
train$Alone <- ifelse(test = train$SibSp == 0 & train$Parch == 0, yes = T, no = F)
test$Alone <- ifelse(test = test$SibSp == 0 & test$Parch == 0, yes = T, no = F)







########### impute age ####################
temp <- rbind(train, test)
imputeAge <- function(boots = 100, in.ratio = 0.66, temp) {
  age.predicts <- c()
  train.temp <- temp[!is.na(temp$Age), ]
  test.temp <- temp[is.na(temp$Age), ]
  
  for(i in 1:boots) {
    s <- sample(x = 1:nrow(train.temp), size = round(x = nrow(train.temp)*in.ratio, digits = 0), replace = T)
    train.current <- train.temp[s, ]
    
    rf.imput <- randomForest(Age~Pclass+Parch+SibSp+Young+Title+Fare+Embarked, data = train.current, ntree = 3000, na.action = na.omit)
    x <- predict(object = rf.imput, newdata = test.temp[, -1], type = "response")
    age.predicts <- rbind(age.predicts, x)
    
    cat("i:", i, "\n")
  }
  return(age.predicts)
}


#updated train
train.temp <- temp[!is.na(temp$Age), ]
test.temp <- temp[is.na(temp$Age), ]
age.predicts <- imputeAge(boots = 100, in.ratio = 0.66, temp = temp)
temp$Age[which(is.na(temp$Age))] <- apply(age.predicts, MARGIN = 2, FUN = median)
train$Age <- temp$Age[1:891]
test$Age <- temp$Age[892:1309]

rm(x)
rm(temp)
rm(rf.imput)
rm(test.temp)
rm(train.temp)
rm(age.predicts)
rm(i)







################# impute cabin ###############

imputeCabin <- function(boots = 100, in.ratio = 0.66, train, test) {
  temp <- rbind(train, test)
  temp <- temp[, c("Cabin", "Pclass", "Fare", "Embarked")]
  temp$Cabin <- as.character(temp$Cabin)
  
  train.temp <- temp[temp$Cabin != "", ]
  test.temp <- temp[temp$Cabin == "", ]
  

  cabin.predicts <- c()
  for(i in 1:boots) {
    s <- sample(x = 1:nrow(train.temp), size = round(x = nrow(train.temp)*in.ratio, digits = 0), replace = T)
    train.current <- train.temp[s, ]
    train.current$Cabin <- as.factor(as.character(train.current$Cabin))
    
    rf.imput <- randomForest(Cabin~., data = train.current, ntree = 2000, na.action = na.omit)
    x <- predict(object = rf.imput, newdata = test.temp[, -1], type = "response")
    cabin.predicts <- rbind(cabin.predicts, as.character(x))
    
    cat("i:", i, "\n")
  }
  
  max.cabin <- apply(cabin.predicts, MARGIN = 2, FUN = count)
  final.cabins <- c()
  for(mc in max.cabin) {
    if(max(mc$freq) >= (0.7*boots)) {
      c <- as.character(mc$x[which(mc$freq >= (0.7*boots))])
    }
    else {
      c <- as.character(mc$x[which(mc$freq == max(mc$freq))[1]])
    }
    
    final.cabins <- c(final.cabins, c)
  }
  
  browser()
  temp$Cabin <- as.character(temp$Cabin)
  temp$Cabin[temp$Cabin == ""] <- final.cabins
  return(temp)
}

test$Fare[153] <- 8
imputed.cabins <- imputeCabin(boots = 200, in.ratio = 0.66, train = train, test = test)
train$Cabin <- imputed.cabins$Cabin[1:891]
test$Cabin <- imputed.cabins$Cabin[892:1309]
rm(imputed.cabins)

train$Cabin <- as.factor(as.character(train$Cabin))
test$Cabin <- as.factor(as.character(test$Cabin))





############# ticked parse ##########
train$Ticket <- as.character(train$Ticket)
test$Ticket <- as.character(test$Ticket)

train$TicketNew <- NA
for(t in 1:nrow(train)) {
  new.ticket <- unlist(strsplit(train$Ticket[t], split = " "))
  train$TicketNew[t] <- new.ticket[1]
}

test$TicketNew <- NA
for(t in 1:nrow(test)) {
  new.ticket <- unlist(strsplit(test$Ticket[t], split = " "))
  
  if(length(new.ticket) > 1) {
    test$TicketNew[t] <- new.ticket[1]
  }
  else {
    test$TicketNew[t] <- "other"
  }
}

rm(new.ticket)
rm(t)







############## impute embarked ##############

imputeEmbarked <- function(boots = 100, in.ratio = 0.66, train, test) {
  temp <- rbind(train, test)
  temp$Embarked <- as.character(temp$Embarked)

  temp <- temp[, c("Embarked", "Cabin", "Alone", "SibSp", "Parch")]
  
  train.temp <- temp[temp$Embarked != "", ]
  test.temp <- temp[temp$Embarked == "", ]
  temp$Embarked <- as.factor(as.character(temp$Embarked))
  
  browser()
  
  embarked.predicts <- c()
  for(i in 1:boots) {
    s <- sample(x = 1:nrow(train.temp), size = round(x = nrow(train.temp)*in.ratio, digits = 0), replace = T)
    train.current <- train.temp[s, ]
    train.current$Embarked <- as.factor(as.character(train.current$Embarked))
    
    rf.imput <- randomForest(Embarked~Cabin+Alone+SibSp+Parch, data = train.current, ntree = 2000, na.action = na.omit)
    x <- predict(object = rf.imput, newdata = test.temp[, -1], type = "response")
    embarked.predicts <- rbind(embarked.predicts, as.character(x))
    
    cat("i:", i, "\n")
  }
  
  browser()
  
  max.embarked <- apply(embarked.predicts, MARGIN = 2, FUN = count)
  final.embarked <- c()
  for(me in max.embarked) {
    if(max(me$freq) >= (0.7*boots)) {
      c <- as.character(me$x[which(me$freq >= (0.7*boots))])
    }
    else {
      c <- as.character(me$x[which(me$freq == max(me$freq))[1]])
    }
    
    final.embarked <- c(final.embarked, c)
  }
  
  browser()
  temp$Embarked <- as.character(temp$Embarked)
  temp$Embarked[temp$Embarked == ""] <- final.embarked
  return(temp)
}

imputed.embarked <- imputeEmbarked(boots = 200, in.ratio = 0.66, train = train, test = test)
train$EmbarkedNew <- imputed.embarked$Embarked[1:891]
test$EmbarkedNew <- imputed.embarked$Embarked[892:1309]
rm(imputed.embarked)

train$EmbarkedNew <- as.factor(as.character(train$EmbarkedNew))
test$EmbarkedNew <- as.factor(as.character(test$EmbarkedNew))






################ relevel ###############
levels(test$Sex) <- levels(train$Sex)
levels(test$cabin.binary) <- levels(train$cabin.binary)
levels(test$Embarked) <- levels(train$Embarked)
levels(test$EmbarkedNew) <- levels(train$EmbarkedNew)
levels(test$CabinDiscrete) <- levels(train$CabinDiscrete)
levels(test$Survived) <- levels(train$Survived)

train$Young <- as.factor(train$Young)
test$Young <- as.factor(test$Young)
levels(test$Young) <- levels(train$Young)
levels(test$Cabin) <- levels(train$Cabin)
levels(train$Title) <- levels(test$Title)
levels(train$Cabin) <- levels(test$Cabin)

levels(test$Alone) <- levels(train$Alone)










################# remove bad cols #################
# train$Name <- NULL
# test$Name <- NULL
# 
# train$Ticket <- NULL
# test$Ticket <- NULL
# 
# train$PassengerId <- NULL
# test$PassengerId <- NULL


save(train, file = "train1.RData")
save(test, file = "test1.RData")


############################ svm #########################

load(file = "train.RData")
load(file = "test.RData")

z <- rbind(train, test)
z$Cabin <- as.factor(as.character(z$Cabin))
z$TicketNew <- as.factor(as.character(z$TicketNew))
z$EmbarkedNew <- as.factor(as.character(z$EmbarkedNew))


# # tune svm params
tuned <- tune.svm(Survived~Title+Sex+Pclass+Age+SibSp+Parch+Embarked+Cabin, data = z[1:891, ], type = "C-classification", gamma = 10^(seq(from = -1, to = 1, by = 0.25)), cost = 10^(seq(from = -1, to = 1, by = 0.25)))
summary(tuned)
# rm(tuned)
# 
# 
# is.numeric(z$TicketNew[2])
# 
# save(train, file = "train.cabin.RData")
# save(test, file = "test.cabin.RData")




# learn
svm <- e1071::svm(Survived~Title+Sex+Pclass+Age+SibSp+Parch+Embarked+Cabin, data = z[1:891, ], type = "C-classification", cost = 0.1778279, gamma = 0.1, kernel="radial") #0.1778279 #0.3162
length(which(as.numeric(as.character(svm$fitted)) != as.numeric(as.character(z$Survived[1:891]))))

# svm <- e1071::svm(Survived~Title+Sex+Pclass+Age+SibSp+Parch+EmbarkedNew+Cabin+TicketNew, data = z[1:891, ], type = "C-classification", cost = 0.17, gamma = 0.1, kernel="radial")
# length(which(as.numeric(as.character(svm$fitted)) != as.numeric(as.character(z$Survived[1:891]))))



predict.svm <- predict(object = svm, newdata = z[892:1309, ], type = "response")
o <- read.csv(file = "0.799.csv")
which(as.numeric(as.character(predict.svm)) != o$Survived)
rm(o)


predict.svm <- predict(object = svm, newdata = z[892:1309, ], type = "response")
l <- read.csv(file = "newest.csv")
which(as.numeric(as.character(predict.svm)) != l$Survived)
rm(l)



# export
export.svm <- data.frame(PassengerId = pass.id, Survived = predict.svm, row.names = NULL)
which(is.na(export.svm$Survived))
write.table(export.svm, file = "newest2.csv", quote = F, sep = ",", row.names = F, col.names = T)




bootstrapSurvived <- function(boots = 100, in.ratio = 0.66, z) {
  predicted.survival <- c()
  for(i in 1:boots) {
    s <- sample(x = 1:891, size = round(x = 891*in.ratio, digits = 0), replace = T)
    
    svm <- e1071::svm(Survived~Title+Sex+Pclass+Age+SibSp+Parch+Embarked+Cabin, data = z[s, ], 
                      type = "C-classification", cost = 0.3162, gamma = 0.1, kernel="radial")
    predict.svm <- predict(object = svm, newdata = z[892:1309, ], type = "response")
    predicted.survival <- rbind(predicted.survival, as.character(predict.svm))
    
    cat("i:", i, "\n")
  }
  
  max.survival <- apply(predicted.survival, MARGIN = 2, FUN = count)
  final.survival <- c()
  final.freq <- c()
  for(mc in max.survival) {
    if(max(mc$freq) >= (0.7*boots)) {
      c <- as.character(mc$x[which(mc$freq >= (0.7*boots))])
    }
    else {
      c <- as.character(mc$x[which(mc$freq == max(mc$freq))[1]])
    }
    
    f <- (max(mc$freq)[1])/boots
    
    final.freq <- c(final.freq, f)
    final.survival <- c(final.survival, c)
  }
  
  result <- list(final.survival = final.survival, final.freq = final.freq)
  
  
  
  return (result)
}



boot.survival <- bootstrapSurvived(boots = 500, in.ratio = 0.66, z)



boot.survival$final.survival
which(as.numeric(boot.survival$final.survival) != o$Survived)
which(as.numeric(boot.survival$final.survival) != l$Survived)

# export
export.svm <- data.frame(PassengerId = pass.id, Survived = as.numeric(boot.survival$final.survival), row.names = NULL)
which(is.na(export.svm$Survived))
write.table(export.svm, file = "newest3.csv", quote = F, sep = ",", row.names = F, col.names = T)





plot(boot.survival$final.freq)
plot(test$Title, boot.survival$final.freq)

plot(test$Title, boot.survival$final.survival)




