library(tidyverse)
library(wordcloud)
library(tm)
library(magrittr)

# load data
raw_df <- read_csv("data/sms_spam.csv")

# Extract label (spam or ham) and convert to factor: 
label <- raw_df$type %>% as.factor

# word cloud for spam

par(bg = "black") 
set.seed(1700)

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

# Preapre data for modelling: 

library(tm)

sms_corpus <- raw_df$text %>% 
  VectorSource() %>% 
  VCorpus() %>% 
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removeNumbers) %>% 
  tm_map(removeWords, stopwords()) %>% 
  tm_map(removePunctuation) %>% 
  tm_map(stripWhitespace)

# Convert to DTM sparse matrix: 
dtm <- sms_corpus %>% DocumentTermMatrix()

# List of words that appear more than 20: 

at_least20 <- findFreqTerms(dtm, 20)

# Convert sparse matrix to data frame: 

inputs <- apply(dtm[, at_least20], 2, 
                function (x) {case_when(x == 0 ~ "No", TRUE ~ "Yes") %>% as.factor()})

df <- inputs %>% 
  as.matrix() %>% 
  as.data.frame() %>% 
  mutate(class = label) 

# split data to train and test
library(caret)
set.seed(1)
train_id <- createDataPartition(df$class, p = 0.7, list = F)
train_df <- df[train_id, ]
test_df <- df[-train_id, ]

# use paralell computing
library(doParallel)
registerDoParallel(cores = detectCores() - 1)

# activate H2o package
library(h2o)
h2o.init()


# convert to h2o frame 
h2o.no_progress()
test <- as.h2o(test_df)
train <- as.h2o(train_df)

# set target and independent variables
y <- 'class'
x <- setdiff(names(train), y)

# Set hyperparameter grid: 

hyper_grid.h2o <- list(ntrees = seq(50, 500, by = 50),
                       mtries = seq(3, 5, by = 1),
                       sample_rate = c(0.55, 0.632, 0.75))

# set random grid search criterias
search_criteria_2 <- list(strategy = "RandomDiscrete",
                          stopping_metric = "AUC",
                          stopping_tolerance = 0.005,
                          stopping_rounds = 10,
                          max_runtime_secs = 30*60)

# train random forest models
system.time(random_grid <- h2o.grid(algorithm = "randomForest",
                                    grid_id = "rf_grid2",
                                    x = x, 
                                    y = y, 
                                    seed = 29, 
                                    nfolds = 10, 
                                    training_frame = train,
                                    hyper_params = hyper_grid.h2o,
                                    search_criteria = search_criteria_2))

# Collect the results and sort by our models: 
grid_perf2 <- h2o.getGrid(grid_id = "rf_grid2", 
                          sort_by = "AUC", 
                          decreasing = FALSE)
# best model
best_model2 <- h2o.getModel(grid_perf2@model_ids[[1]])


library(pROC)
# Function calculates AUC: 

auc_for_test <- function(model_selected) {
  actual <- test_df$class
  pred_prob <- h2o.predict(model_selected, test) %>% as.data.frame() %>% pull(spam)
  return(roc(actual, pred_prob))
}

# Use this function: 
my_auc <- auc_for_test(best_model2)


