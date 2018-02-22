rm(list = ls())
gc()

require(tidyverse)
require(caret)
require(data.table)
require(lubridate)

## reading train data set
train_data <- fread('..\\data\\train.csv', data.table = F)
train_data$date <- ymd(train_data$date)
train_data <- train_data %>%
  mutate(year = year(date), month = month(date), day = day(date),
         weekday = wday(date), week_of_year = week(date))
glimpse(train_data)

# convert -ve sales to zero and take log of it 
# log is taken as evaluation metric is log based and because sales data is right skewed 
train_data$unit_sales[train_data$unit_sales < 0] <- 0
train_data$unit_sales <- log1p(train_data$unit_sales)


# Reading other data files -----------------------------------------

## all items data set
items <- fread('..\\data\\items.csv', data.table = F)
glimpse(items)
summary(items)

## oil prices 
oil <- fread('..\\data\\oil.csv', data.table = F)
glimpse(oil)

## holidays events
holidays_events <- fread('..\\data\\holidays_events.csv', data.table = F)
glimpse(holidays_events)
str(holidays_events)
holidays_events$date <- ymd(holidays_events$date)
char_vars <- c("type", "locale", "locale_name")
holidays_events <- map_at(holidays_events, char_vars, as.factor) %>%  
  tibble::as_tibble(.)
head(holidays_events)
summary(holidays_events)
holidays_events %>% group_by(type) %>% count() %>% arrange(desc(n))


# Analysing data ----------------------

# count of perishable items
items %>% group_by(perishable) %>% count()

# top 20 item families
items %>% group_by(family) %>% 
  summarise(count = n()) %>%
  top_n(20) %>%
  arrange(desc(count)) %>%
  ggplot() + geom_col(aes(x = reorder(family,count), y = count), fill = 'red3') + coord_flip()

# top item families and their class
items %>% group_by(class, family) %>% 
  summarise(count = n()) %>%
  arrange(desc(count))



# Joining item data with train data for further analysis ---------------------

train_data <- train_data %>%
  left_join(items) %>%
  left_join(holidays_events) 

head(train_data)
glimpse(train_data)

# difference between holiday and non holiday sales
train_data <- train_data %>%
  mutate(type = as.factor(ifelse(is.na(locale),"no holiday", "holiday")))

# variation of sales by holiday type
train_data %>% 
  group_by(type) %>% 
  summarise(avg_sales = mean(unit_sales)) %>%
  ggplot() + geom_col(aes(x = type, y = avg_sales))

# perishable vs non perishable sales
train_data %>% 
  group_by(perishable) %>% 
  summarise(total_sales = sum(unit_sales)) 

# promotion vs non promotion avg sales
train_data %>% 
  group_by(onpromotion) %>% 
  summarise(avg_sales = mean(unit_sales)) %>%
  ggplot() + geom_col(aes(x = onpromotion, y = avg_sales))

## promotion and holidays have significant impact on sales
  

## Analyzing transactions and store data ------------------------------------------
transactions <- fread('transactions.csv', data.table = F)
glimpse(transactions)
summary(transactions)

stores <- fread('stores.csv', data.table = F)
glimpse(stores)

# Joining stores and transactions data for analysis
transactions <- transactions %>% left_join(stores) 
head(transactions)

# convert character features to factors for quick analysis
transactions$date <- ymd(transactions$date)
char_vars <- c("city", "state","type","cluster")
transactions <- map_at(transactions, char_vars, as.factor) %>%  
  tibble::as_tibble(.)
head(transactions)

# transactions by city
transactions %>% 
  group_by(city)%>% 
  summarise(total_trans = sum(transactions)) %>% 
  arrange(desc(total_trans)) %>%
  ggplot() + geom_col(aes(x = reorder(city,-total_trans), y = total_trans), fill = "seagreen") + theme_bw()

# transactions by store cluster
transactions %>% 
  group_by(cluster)%>% 
  summarise(total_trans = sum(transactions)) %>% 
  arrange(desc(total_trans)) %>%
  ggplot() + geom_col(aes(x = reorder(cluster,-total_trans), y = total_trans), fill = "red") 

# transactions by store type
transactions %>% 
  group_by(type)%>% 
  summarise(total_trans = sum(transactions)) %>% 
  arrange(desc(total_trans)) %>%
  ggplot() + geom_col(aes(x = reorder(type,-total_trans), y = total_trans)) +   theme_minimal()

# transactions by store
total_trans <- sum(transactions$transactions)
store_trans <- transactions %>% 
  select(store_nbr, transactions, type) %>%
  group_by(store_nbr, type)%>% 
  summarise(total_trans = (sum(transactions)*100)/total_trans) %>% 
  arrange(desc(total_trans)) %>%
  mutate(cumsum = cumsum(total_trans)) 

print(store_trans %>% filter, n=35)

ggplot() + 
  geom_col(aes(x = reorder(store_nbr,-total_trans), y = total_trans, fill = type)) +   
  theme_minimal()


transactions %>% 
  ggplot() +  
  geom_histogram(aes(log(transactions)), color = "seagreen", alpha = 0.25)  
  


# top 20 items by sales
train_data %>% 
  group_by(item_nbr)%>% 
  summarise(total_sales = sum(unit_sales)) %>% 
  arrange(desc(total_sales)) %>%
  top_n(20) %>%
  ggplot() + geom_col(aes(x = reorder(item_nbr,-total_sales), y = total_sales), fill = "red") 

# % sales contribution by various items
total_sales <- sum(train_data$unit_sales)
item_sales <- train_data %>% 
  group_by(item_nbr)%>% 
  summarise(perc_sales = (sum(unit_sales)*100)/total_sales) %>% 
  arrange(desc(perc_sales)) %>%
  ungroup() %>%
  mutate(cumsum_sales = cumsum(perc_sales)) 

item_sales[item_sales$perc_sales > 0.02,]
print(item_sales %>% filter, n=35)

train_data %>% 
  group_by(item_nbr)%>% 
  summarise(total_sales = sum(unit_sales)) %>% 
  arrange(desc(-total_sales)) 

train_data %>% 
  group_by(item_nbr)%>% 
  summarise(total_sales = sum(unit_sales)) %>% 
  ggplot() + geom_histogram(aes(x = total_sales))



holiday_store <- fread('store_holiday.csv')
holiday_store$date <- ymd(holiday_store$date)
train_data <- train_data %>%
  left_join(holiday_store, by = c('store_nbr', 'date'))
glimpse(train_data)
summary(train_data$holiday)

train_data$holiday[is.na(train_data$holiday)] <- FALSE

# sales by day of the month
train_data %>% 
  # filter(month >= 10)%>%
  # filter(day > 15)%>%
  group_by(day)%>% 
  summarise(avg_sales = mean(expm1(unit_sales))) %>% 
  ggplot() + geom_col(aes(x = day, y = avg_sales),fill = 'dodgerblue2') +
  theme_minimal()

# analysing Aug sales data as it look different from other months
train_data %>% 
  filter(month == 8 & holiday == FALSE)%>%
  # filter(day > 15)%>%
  group_by(day)%>% 
  summarise(avg_sales = mean(expm1(unit_sales))) %>% 
  ggplot() + geom_col(aes(x = day, y = avg_sales),fill = 'dodgerblue2') +
  + ggtitle("Aug sales w/o holidays") +
  xlab("day") + ylab("avg sales (log)")


# plot July to Sep data for comparison with Aug
train_data %>%
  filter(month >= 7 & month <= 9)%>%
  filter(!is.na(year)) %>%
  filter(!is.na(month)) %>%
  group_by(day,month) %>%
  summarise(Count = sum(unit_sales)) %>%
  mutate(DayMonth = make_date(day=day,month=month)) %>%
  ggplot(aes(x=DayMonth,y=Count,group = 1)) +
  geom_line(size=1, color="red")+
  geom_point(size=3, color="red") +
  labs(x = 'Time', y = 'Count',title = 'Trend of Sales') +
  theme_bw() 

# monthly on promotion vs non promotion sales
train_data %>% 
  group_by(month,year, onpromotion)%>%
  summarise(avg_sales = mean(unit_sales)) %>%
  ggplot() + geom_boxplot(aes(x = onpromotion, y = avg_sales))

# weekly sales trend
train_data %>% 
  group_by(weekday)%>%
  summarise(avg_sales = mean(unit_sales)) %>%
  ggplot() + geom_col(aes(x = weekday, y = avg_sales),fill = 'dodgerblue2') +
  + ggtitle("weekly sales") +
  xlab("weekday") + ylab("avg sales (log)")


# Analysis of top item families and item type
train_data$family <- as.factor(train_data$family)

train_data %>%
  group_by(family) %>%
  summarise(sum_family = sum(unit_sales)) %>%
  arrange(desc(sum_family)) %>%
  top_n(20) %>%
  ggplot() + 
  geom_col(aes(x = reorder(family,-sum_family), y = sum_family), fill = 'dodgerblue2') +
  theme_minimal()

train_data %>%
  group_by(type) %>%
  summarise(avg_sales = mean(unit_sales)) %>%
  ggplot() + 
  geom_col(aes(x = type, y = avg_sales), fill = 'red4') +
  theme_minimal()
