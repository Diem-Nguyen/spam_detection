library(tidyverse)
library(wordcloud)
library(tm)

# load data
raw_df <- sms_spam <- read_csv("data/sms_spam.csv")

# convert type to factor
raw_df$type <- raw_df$type %>% as.factor

# word cloud for spam

par(bg = "black") 
set.seed(1709)

word <- wordcloud(raw_df  %>% filter(type == "spam") %>% pull(text), 
          max.words = 150, 
          random.order = FALSE, 
          rot.per = 0.35, 
          font = 2,
          colors = brewer.pal(8, "Dark2"))

# save as image
png("word.png")
print(word)     
dev.off() 

# prepare data for modeling

sms_corpus <- raw_df$text %>% 
  VectorSource() %>% 
  VCorpus() %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace)

# convert to sparse matrix 

