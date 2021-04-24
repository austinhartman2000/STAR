library(readr)
library(rpart)

data <- read_csv("The Model Folder/modelFinalDataCategories-current.csv")
data2 = data[which(data$Year != 2019),]
# Random Forest prediction of Kyphosis data
install.packages('randomForest')
library(randomForest)
nums = c()
for (i in 3:113){
  if((i-3)%%11 <= 9){
    nums = append(nums, i)
  }
}
fit <- randomForest(SRS~., data2[,nums])
print(fit) # view results
importance(fit) # importance of each predictor
data$rfpredict = predict(fit, data)
plot(data$rfpredict, data$SRS)
lm3 = lm(SRS ~ rfpredict, data)
summary(lm3)

data3 = data[which(data$Year == 2019),]
plot(data3$rfpredict, data3$SRS, col = "white", pch = 1, cex = 2)
with(data3, text(SRS~rfpredict, labels = (data3$Team), font = 1, cex = 0.9))
lm4 = lm(SRS ~ rfpredict, data3)
abline(lm4)
summary(lm4)
print(cor(data3$rfpredict, data3$SRS))
print(sqrt(sum((data3$SRS-data3$rfpredict)**2)/nrow(data3)))
print(sum(abs(data3$SRS-data3$rfpredict))/nrow(data3))

library(dplyr)
View(select(data3, 'Team', 'rfpredict'))
#____________________________________________________________________________________________
archetypes = read_csv("The Model Folder/archetypes3.csv")
archetypes = archetypes[which(archetypes$Year == 2019),]
team = "TOR"
trade_player = "Fred VanVleet"
trade_player_vorp = 0
team_year = 2019
player = ""
preds = c()
vorps = c()
year = 2019
for (row in 1:nrow(archetypes)){
  player_name = archetypes$Player[row]
  player_arc = archetypes$Archetype[row]
  player_a = archetypes$A[row]
  player_b = archetypes$B[row]
  player_c = archetypes$C[row]
  player_d = archetypes$D[row]
  player_e = archetypes$E[row]
  player_impact = archetypes$VORP[row]
  player_O = archetypes$OBPM[row]
  player_D = archetypes$DBPM[row]
  player_MP = archetypes$MP[row]
  mavs = data[which(data$Team == team & data$Year == team_year),]
  prediction = predict(fit,mavs)
  start_point = 0
  for (i in 0:9){
    if(trade_player == mavs[[paste('Name', i, sep = '')]]){
      trade_player_vorp = mavs[[paste('VORP', i, sep = '')]]
      mavs[[paste('Name', i, sep = '')]] = player_name
      mavs[[paste('Arch', i, sep = '')]] = player_arc
      mavs[[paste('A', i, sep = '')]] = player_a
      mavs[[paste('B', i, sep = '')]] = player_b
      mavs[[paste('C', i, sep = '')]] = player_c
      mavs[[paste('D', i, sep = '')]] = player_d
      mavs[[paste('E', i, sep = '')]] = player_e
      mavs[[paste('OBPM', i, sep = '')]] = player_O
      mavs[[paste('DBPM', i, sep = '')]] = player_D
      mavs[[paste('VORP', i, sep = '')]] = player_impact
      mavs[[paste('MP', i, sep = '')]] = player_MP
      start_point = i
      break
    }
  }
  stop_point = 0
  for (i in 0:9){
    if(player_impact > mavs[[paste('VORP', i, sep = '')]] && player_name != mavs[[paste('Name', i, sep = '')]]){
      stop_point = stop_point + 1
    }
    
  }
  #print(c(mavs$Name0, mavs$VORP0, mavs$Name1, mavs$VORP1, mavs$Name2, mavs$VORP2, mavs$Name3, mavs$VORP3, mavs$Name4, mavs$VORP4))
  stop_point= 9-stop_point
  #print(c(start_point, stop_point))
  if(start_point<stop_point){
    for(j in start_point:(stop_point-1)){
      #print(j)
      mavs[[paste('Name', j, sep = '')]] = mavs[[paste('Name', j+1, sep = '')]]
      mavs[[paste('Arch', j, sep = '')]] = mavs[[paste('Arch', j+1, sep = '')]]
      mavs[[paste('A', j, sep = '')]] = mavs[[paste('A', j+1, sep = '')]]
      mavs[[paste('B', j, sep = '')]] = mavs[[paste('B', j+1, sep = '')]]
      mavs[[paste('C', j, sep = '')]] = mavs[[paste('C', j+1, sep = '')]]
      mavs[[paste('D', j, sep = '')]] = mavs[[paste('D', j+1, sep = '')]]
      mavs[[paste('E', j, sep = '')]] = mavs[[paste('E', j+1, sep = '')]]
      mavs[[paste('OBPM', j, sep = '')]] = mavs[[paste('OBPM', j+1, sep = '')]]
      mavs[[paste('DBPM', j, sep = '')]] = mavs[[paste('DBPM', j+1, sep = '')]]
      mavs[[paste('VORP', j, sep = '')]] = mavs[[paste('VORP', j+1, sep = '')]]
      mavs[[paste('MP', j, sep = '')]] = mavs[[paste('MP', j+1, sep = '')]]
    }
  }
  if(start_point > stop_point){
    for(j in start_point:(start_point+1)){
      mavs[[paste('Name', j, sep = '')]] = mavs[[paste('Name', j-1, sep = '')]]
      mavs[[paste('Arch', j, sep = '')]] = mavs[[paste('Arch', j-1, sep = '')]]
      mavs[[paste('A', j, sep = '')]] = mavs[[paste('A', j-1, sep = '')]]
      mavs[[paste('B', j, sep = '')]] = mavs[[paste('B', j-1, sep = '')]]
      mavs[[paste('C', j, sep = '')]] = mavs[[paste('C', j-1, sep = '')]]
      mavs[[paste('D', j, sep = '')]] = mavs[[paste('D', j-1, sep = '')]]
      mavs[[paste('E', j, sep = '')]] = mavs[[paste('E', j-1, sep = '')]]
      mavs[[paste('OBPM', j, sep = '')]] = mavs[[paste('OBPM', j-1, sep = '')]]
      mavs[[paste('DBPM', j, sep = '')]] = mavs[[paste('DBPM', j-1, sep = '')]]
      mavs[[paste('VORP', j, sep = '')]] = mavs[[paste('VORP', j-1, sep = '')]]
      mavs[[paste('MP', j, sep = '')]] = mavs[[paste('MP', j-1, sep = '')]] 
    }
  }
  mavs[[paste('Name', stop_point, sep = '')]] = player_name
  mavs[[paste('Arch', stop_point, sep = '')]] = player_arc
  mavs[[paste('A', stop_point, sep = '')]] = player_a
  mavs[[paste('B', stop_point, sep = '')]] = player_b
  mavs[[paste('C', stop_point, sep = '')]] = player_c
  mavs[[paste('D', stop_point, sep = '')]] = player_d
  mavs[[paste('E', stop_point, sep = '')]] = player_e
  mavs[[paste('OBPM', stop_point, sep = '')]] = player_O
  mavs[[paste('DBPM', stop_point, sep = '')]] = player_D
  mavs[[paste('VORP', stop_point, sep = '')]] = player_impact
  mavs[[paste('MP', stop_point, sep = '')]] = player_MP
  #print(c(mavs$Name0, mavs$VORP0, mavs$Name1, mavs$VORP1, mavs$Name2, mavs$VORP2, mavs$Name3, mavs$VORP3, mavs$Name4, mavs$VORP4))
  #print("----------------------------------------")
  prediction2 = predict(fit, mavs)
  preds = append(preds,prediction2)
  vorps = append(vorps, player_impact)
}
options(scipen = 999)
finaldata = data.frame(archetypes$Player, preds, vorps)
finaldata = finaldata[order(-preds),]
finaldata$Rank = 1:(nrow(finaldata))
View(finaldata)
mavs = data[which(data$Team == team & data$Year == team_year),]
prediction = predict(fit,mavs)
print(c(team, trade_player, trade_player_vorp))
print(prediction)