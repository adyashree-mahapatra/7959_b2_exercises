library(readxl)
library(dplyr)

excel_sheets("SaleData.xlsx")
df <- read_excel("SaleData.xlsx", sheet = "Sales Data")
diamond<-read.csv("diamonds.csv")
movie<-read.csv('movie_metadata.csv')
imdb<-read.csv('imdb1.csv')

#Question-1
q1<-function(df){
  ans<-df %>% group_by(Item) %>% summarise(min_sales = min(Sale_amt))
  return(ans)
}
q1(df)

# Question-2
q2 <-function(df){
  ans<-df %>% group_by(format(as.Date(OrderDate,format="%Y-%m-%d"),"%Y"),Region) %>% summarise(total_sales = sum(Sale_amt))
  return(ans)
}
q2(df)

#Question-3
q3<-function(df){
  df$days_diff <- Sys.Date()- as.Date(df$OrderDate,format="%Y-%m-%d")
  return(head(df))
}
q3(df)

#Question-4
q4<-function(df){
  data1 <- data.frame(manager=df$Manager,list_of_salesmen=df$SalesMan)
  ans<-data1 %>% group_by(manager) %>% summarise(list_of_salesmen = paste(unique(list_of_salesmen),collapse = ","))
  return(ans)
}
q4(df)

#Question-5
q5<-function(df){
  data2 <- data.frame(Region=df$Region,Salesmen_count=df$SalesMan, total_sales=df$Sale_amt)
  data3 <- data2 %>% group_by(Region) %>%  summarise(total_sales= sum(total_sales))
  data4 <- data2 %>% group_by(Region) %>% count(Salesmen_count) %>% count(Region)
  ans<- data.frame(data3,Salesmen_count=data4$n)
  return(ans)
}
q5(df)

#Question-6
q6<-function(df){
  data6 <- data.frame(Manager=df$Manager, Total_sale=df$Sale_amt)
  total_sale_amount= sum(df$Sale_amt)
  ans<- data6 %>% group_by(Manager) %>% summarise(percent_sales= sum(Total_sale)*100/total_sale_amount)
  return(ans)
}
q6(df)

#Question-7

q7<-function(df){
  ans<-df[5,6]
  return(ans)
}
q7(imdb)

#Question-8

q8<-function(df){
  ma<-which.max(df$duration)
  mi<-which.min(df$duration)
  r1<-as.numeric(ma)
  r2<-as.numeric(mi)
  df$duration<-as.numeric(as.character(df$duration))
  print("title:")
  print(df[r1,3])
  print("duration:")
  print(df[r1,8])
  print("title:")
  print(df[r2,3])
  print("duration:")
  print(df[r2,8])
}
q8(imdb)

#Question-9
q9<-function(df){
  df$imdbRating<-as.numeric(as.character(df$imdbRating))
  f <-df[order(df$year,-df$imdbRating),]
  return(f)
}
q9(imdb)

#Question-10
q10<-function(df){
  newdata <- subset(df, gross>2000000)
  newdata1 <- subset(newdata, budget<1000000)
  newdata2 <- subset(newdata1, duration >= 30 & duration < 180)
  return(newdata2)
}
q10(movie)



#Question-11
q11<-function(df){
  r<-(nrow(df)-nrow(distinct(df)))
  return(r)
}
q11(diamond)

#Question-12
q12<-function(df){
  df <- df[-which(df$carat == ""), ]
  df <- df[-which(df$cut == ""), ]
  return(df)
}
q12(diamond)

#Question-13
q13<-function(df){
  r<-select_if(df,is.numeric)
  return(r)
}
q13(diamond)

diamond$z<-as.numeric(as.character(diamond$z))
cbind(diamond,volume=0)

#Question-14

q14<-function(df){
  i<-1
  x<-nrow(df)
  while(i<=x)
  {
    if(df[i,5]>60){
      df[i,"volume"]<-df[i,8]*df[i,9]*df[i,10]
    } else{
      df[i,"volume"]<-8
    }
    i=i+1
  }
  return(df)
}
q14(diamond)

#Question-15
q15<-function(df){
  i<-1
  df[is.na(df)]<-0
  m<-mean(df$price)
  x<-nrow(df)
  while(i<=x){
    if((df[i,7])==0){
      df[i,7]<-m
    }
    i=i+1
  }
  return(df)
}
q15(diamond)

