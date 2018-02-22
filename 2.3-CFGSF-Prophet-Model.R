require(tidyverse)
require(data.table)
require(foreach)
require(lubridate)
require(parallel)
require(doSNOW)
require(Metrics)
require(prophet)


## reading data file and prepating training and validation set
train <- fread('train.csv', skip = 36458909, 
               col.names = c('id', 'date','store_nbr', 'item_nbr', 'unit_sales', 'onpromotion')
)
test <-fread('test.csv')

train[, ':='(
  date = ymd(date, tz = NULL),
  store_item = paste(store_nbr, item_nbr, sep="_")
)]

# transform to log1p
train$unit_sales[train$unit_sales < 0] <- 0
train$unit_sales <- log1p(train$unit_sales)

test[, ':='(
  date = ymd(date, tz = NULL),
  store_item = paste(store_nbr, item_nbr, sep="_")
)]

train <- train[date >= ymd("2015-08-01"),]
train_wide <- dcast(train, store_item ~ date, value.var = "unit_sales", fill = 0)
train <- train_wide
# removing validation part to speed up
# train <- train_wide[, 1:which(colnames(train_wide)=="2017-07-25")]
# valid <- data.table(store_item = train_wide$store_item, train_wide[, which(colnames(train_wide)=="2017-07-26"):ncol(train_wide)])

test_id <- test$id
test <- test[, .(date, store_item)]

# glimpse(train)
# glimpse(valid)
rm(train_wide)

# holiday list as per stores (ran Pedro's script to pull holiday list by store)
store_holidays <- fread('store_holiday.csv', data.table = F)
store_holidays$date <- ymd(store_holidays$date)
store_holidays <- store_holidays %>% filter(date >= "2014-08-01")
glimpse(store_holidays)
stores <- fread('stores.csv', data.table = F)
stores <- stores[,c('store_nbr','city')]
store_holidays <- store_holidays %>% left_join(stores)
# holiday by city
store_holidays <- store_holidays[,c('store_nbr','date','city')]
colnames(store_holidays) <- c('store_nbr','ds','holiday')
head(store_holidays)



# # define a new function which will be called by foreach loop
prophet_model <- function() {
  train_ts <- melt(train[i,], id.vars = c("store_item"))
  s_i_key <- train_ts$store_item[1]
  train_ts$store_item <- NULL
  colnames(train_ts) <- c("ds", "y")
  store_key <- unlist(str_split(s_i_key, "_"))[1]
  holidays <- store_holidays %>% filter(store_nbr == store_key)
  holidays$store_nbr <- NULL
  if (nrow(train_ts) > 2) {
    m <- prophet(train_ts, yearly.seasonality=TRUE, weekly.seasonality = T,
                 daily.seasonality=F)
    future <- make_future_dataframe(m, periods = 16)
    forecast <- predict(m, future)
    forecast <- forecast[c('ds', 'yhat')]
    forecast$store_item <- s_i_key
    # retain only predicted part to reduce memory usage
    forecast <- forecast[forecast$ds >= ymd("2017-08-16"),]
    return(forecast)
  }
}

results <- data.frame(NULL)
# n_cores <- detectCores()
cl <- makeCluster(40L)
registerDoSNOW(cl)
start_p <- Sys.time()
number_of_iterations <- nrow(train)
results <- foreach(i = 1:number_of_iterations, 
                   .packages = c("prophet", 'data.table', 'lubridate', 'tidyverse'), 
                   .combine='rbind') %dopar% {
                     results <- prophet_model()  
                     results
                   }
end_p <- Sys.time()
difftime(end_p, start_p, unit = "min")
stopCluster(cl)
head(results)
tail(results)

# generate the final submission file
submission <- results[,c('store_item', 'ds', 'yhat')]
colnames(submission) <- c('store_item', 'date', 'unit_sales')
test <- test %>% left_join(submission, by = c('store_item', 'date'))
test$unit_sales[test$unit_sales < 0] <- 0
test$unit_sales[is.na(test$unit_sales)] <- 0
test$unit_sales <- expm1(test$unit_sales)
test <- test %>% 
  separate(store_item, into = c('store_nbr', 'item_nbr'), sep = "_")
test$id <- test_id
test <- test[,c('id', 'unit_sales')]
head(test)

# checking % of zeros in the test file
length(test$unit_sales[test$unit_sales == 0])/length(test$unit_sales)
# write file
fwrite(test, "pred_prophet.csv")
