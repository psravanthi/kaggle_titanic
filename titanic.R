setwd("G:/Titanic")
# missForest package used for missing value imputation using random forests
# Caret package used for model building (model implementation, cross validation)
install.packages("missForest")
install.packages("caret")
library("missForest")
library("caret")

train<- read.csv("G:/Marketing Analytics/train.csv", stringsAsFactors =FALSE,na.strings =c(" ",NA,""))
test <- read.csv("test_titanic.csv", stringsAsFactors =FALSE,na.strings =c(" ",NA,""))
test$Survived <- -1

#Combining train and test
combined <- rbind(train,test)
combined$Fare <- ifelse(combined$Fare==0,NA,combined$Fare)

# Extracting title from names and grouping similar titles
combined$title <- sapply(combined$Name, FUN  = function(x) {strsplit(x,split='[,.]')[[1]][2]})
combined$title <- sub(' ','',combined$title)
combined$title[combined$title %in% c('Mme','Mlle')] <- 'Mlle'
combined$title[combined$title %in% c('Dona', 'Lady', 'the Countess')] <- 'Lady'
combined$title[combined$title %in% c('Capt', 'Don', 'Major', 'Sir','Jonkheer','Col')] <- 'Sir'
combined$title <- factor(combined$title)

#Converting variables to factors
combined$Sex <- as.factor(combined$Sex)
combined$Embarked <- as.factor(combined$Embarked)
combined$title <- as.factor(combined$title)

combined$familysize <- combined$SibSp+combined$Parch+1

# Calculating fare per passenger from the grouped ticket prices
tickets <- table(combined$Ticket)
ticketfreq <-  as.data.frame(tickets)
colnames(ticketfreq) <- c("Ticket","NumP")
combined<- merge(combined,ticketfreq,by="Ticket")
combined$FarePerPgr <- combined$Fare/combined$NumP
combined <- combined[order(combined$PassengerId),]

#Removing irrelevant columns
combined_missing <- combined[,-which(names(combined) %in% c("Ticket","Name","SibSp","Parch","Fare","Cabin","NumP"))]

# Multiple imputation using random forests
combined_final <- missForest(combined_missing,ntree=100,verbose=TRUE)
combined_impute <- combined_final$ximp

# Splitting train and test sets
trainf<- combined_impute[combined_impute$Survived != -1,]
trainf$Survived <- as.factor(trainf$Survived)
testf <- combined_impute[combined_impute$Survived==-1,]
testf$Survived <- NULL

# Model building (gbm,random forest) using 10-fold cross validation
trainctrl <- trainControl(method="repeatedcv",verboseIter=TRUE,savePredictions = "final")
modelfit_gbm <- train(Survived~ Pclass+Sex+ Age+Embarked+title+familysize+FarePerPgr,data=trainf,method="gbm",trControl=trainctrl)
modelfit_knn <- train(Survived~ Pclass+Sex+ Age+Embarked+title+familysize+FarePerPgr,data=trainf,method="knn",trControl=trainctrl)
modelfit_rf <- train(Survived~ Pclass+Sex+ Age+Embarked+title+familysize+FarePerPgr,data=trainf,method="rf",trControl=trainctrl)

# Predicting on test data using gbm
test_gbm <- as.data.frame(predict(modelfit_gbm,testf[,-which(names(testf) %in% c("Ticket","Name","SibSp","Parch","Fare","Cabin","NumP"))]))
test_gbm <- cbind(test$PassengerId,test_gbm)
colnames(test_gbm) <- c("PassengerId","Survived")
write.csv(test_gbm,"titanic_gbm_model.csv",row.names=FALSE)

# Predicting on test data using random forest
test_rf <- as.data.frame(predict(modelfit_rf,testf[,-which(names(testf) %in% c("Ticket","Name","SibSp","Parch","Fare","Cabin","NumP"))]))
test_rf <- cbind(test$PassengerId,test_rf)
colnames(test_rf) <- c("PassengerId","Survived")
write.csv(test_rf,"titanic_rf_model.csv",row.names=FALSE)



