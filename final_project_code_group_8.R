#loading required libraries

library(tidyverse)
library(dplyr)
library(tm)
library(text2vec)
library(SnowballC)
library(glmnet)
library(caret)
library(tree)
library(randomForest)
library(xgboost)
library(ggplot2)


#Reading the data

setwd("C:/Users/sreej/OneDrive/Documents/spring 2024/DATA MINING/project/")

train_x <- read_csv('airbnb_train_x_2024.csv')%>%
  mutate(id=row_number())

test_x <- read_csv('airbnb_test_x_2024.csv')%>%
  mutate(id = row_number())

train_y <- read_csv('airbnb_train_y_2024.csv')%>%
  select(perfect_rating_score) %>%
  mutate(perfect_rating_score = factor(ifelse(perfect_rating_score == "YES", 1, 0), levels = c(0, 1)))

##Data cleaning

#Numerical data cleaning

train_1 <- train_x %>%
  
  select(-name,-summary,-space,-description,-experiences_offered,-neighborhood_overview,-notes,-transit,-access,-interaction,-house_rules,
         -host_name,-host_location,-host_about,-host_neighbourhood,-host_verifications,-street,-neighborhood,-neighborhood_group,
         -zipcode,-smart_location,-country,-country_code,-latitude,-longitude,-license,-jurisdiction_names,-first_review )%>%
  
  mutate(host_response_time = as.factor(ifelse(is.na(host_response_time), 'Missing', host_response_time)),
         host_response = as.factor(ifelse(is.na(host_response_rate),"MISSING",ifelse(host_response_rate == 100,"ALL","SOME"))),
         host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate),"MISSING",ifelse(host_acceptance_rate == "100%","ALL","SOME"))),
         host_total_listings_count = ifelse(is.na(host_total_listings_count),host_listings_count,host_total_listings_count),
         market = factor(ifelse(is.na(market)|market %in% names(table(market)[table(market)<300]),"OTHER",market)),
         property_category = ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "Apartment", 
                                    ifelse(property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", 
                                           ifelse(property_type %in% c("Townhouse", "Condominium"), "condo", 
                                                  ifelse(property_type %in% c("Bungalow", "House"), "house", "other")))),
         property_category = as.factor(property_category),
         room_type = as.factor(room_type),
         accommodates = ifelse(is.na(accommodates),ifelse(bedrooms == 0, 1, bedrooms),accommodates),
         bedrooms = ifelse(is.na(bedrooms),ceiling(accommodates/2),bedrooms),
         beds = ifelse(is.na(beds),bedrooms,beds),
         bathrooms = ifelse(is.na(bathrooms),median(bathrooms,na.rm = TRUE),bathrooms),
         bed_category = factor(ifelse(bed_type == 'Real Bed', 'bed', 'other')),
         cancellation_policy = factor(ifelse(cancellation_policy %in%  c('super_strict_30','super_strict_60'),'super_strict',cancellation_policy))
  ) %>%
  group_by(state,city,property_category,room_type) %>%
  mutate(avg_square_feet = mean(square_feet,na.rm=TRUE),
         avg_price = mean(price,na.rm=TRUE))%>%
  ungroup() %>%
  mutate(
    square_feet = ifelse(is.na(square_feet),avg_square_feet,square_feet),
    square_feet = ifelse(is.na(square_feet),median(square_feet,na.rm= TRUE),square_feet),
    price = ifelse(is.na(price),avg_price,price),
    price = ifelse(is.na(price),median(price,na.rm=TRUE),price),
    weekly_price = ifelse(is.na(weekly_price),7*price,weekly_price),
    monthly_price = ifelse(is.na(monthly_price),31*price,monthly_price),
    security_deposit = ifelse(is.na(security_deposit),0,security_deposit),
    cleaning_fee = ifelse(is.na(cleaning_fee),0,cleaning_fee)) %>%
  select(-city,-state,-host_acceptance_rate,-host_response_rate,-property_type,-bed_type,-avg_square_feet,-avg_price,-amenities) %>%
  mutate(
    available_30 = as.factor(cut(availability_30,breaks = c(-Inf, 10, 20, 30),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_60 = as.factor(cut(availability_60,breaks = c(-Inf, 20, 40, 60),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_90 = as.factor(cut(availability_90,breaks = c(-Inf, 30, 60, 90),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_365 = as.factor(cut(availability_365,breaks = c(-Inf, 122, 243, 365),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    features_list = strsplit(features, ","),
    price_per_person = ifelse(is.na(price),0,price/accommodates)) %>%
  group_by(property_category)%>%
  mutate(
    median_ppp = median(price_per_person, na.rm = TRUE)) %>%
  ungroup()%>%
  mutate(
    ppp_ind = ifelse(price_per_person > median_ppp, 1, 0),
    ppp_ind = as.factor(ppp_ind)
  )%>%
  mutate(
    charges_for_extra = as.factor(ifelse(is.na(extra_people) | extra_people == 0,0,1)),
    has_min_nights = as.factor(ifelse(minimum_nights > 1, 1,0)),
    has_cleaning_fee = as.factor(ifelse(as.numeric(cleaning_fee) ==0 | is.na(cleaning_fee) ,0,1)),
    has_security_deposit = as.factor(ifelse(as.numeric(security_deposit) ==0 |is.na(security_deposit) ,0,1))
  )


unique_features <- unique(unlist(train_1$features_list))


for (feature in unique_features) {
  train_1 <- train_1 %>%
    mutate(!!feature := as.integer(sapply(features_list, function(x) feature %in% x)))
}

train_numeric <- train_1 %>%
  select(-features_list,-features,-'NA',-host_since) 


#creating dummies for factor variables
train_dummies <- model.matrix(~ . - 1, data = train_numeric)
train_numeric_final <- data.frame(train_dummies)


#test numerical data cleaning

test_1 <- test_x %>%
  
  select(-name,-summary,-space,-description,-experiences_offered,-neighborhood_overview,-notes,-transit,-access,-interaction,-house_rules,
         -host_name,-host_location,-host_about,-host_neighbourhood,-host_verifications,-street,-neighborhood,-neighborhood_group,
         -zipcode,-smart_location,-country,-country_code,-latitude,-longitude,-license,-jurisdiction_names,-first_review )%>%
  
  mutate(host_response_time = as.factor(ifelse(is.na(host_response_time), 'Missing', host_response_time)),
         host_response = as.factor(ifelse(is.na(host_response_rate),"MISSING",ifelse(host_response_rate == 100,"ALL","SOME"))),
         host_acceptance = as.factor(ifelse(is.na(host_acceptance_rate),"MISSING",ifelse(host_acceptance_rate == "100%","ALL","SOME"))),
         host_total_listings_count = ifelse(is.na(host_total_listings_count),host_listings_count,host_total_listings_count),
         market = factor(ifelse(is.na(market)|market %in% names(table(market)[table(market)<300]),"OTHER",market)),
         property_category = ifelse(property_type %in% c("Apartment", "Serviced apartment", "Loft"), "Apartment", 
                                    ifelse(property_type %in% c("Bed & Breakfast", "Boutique hotel", "Hostel"), "hotel", 
                                           ifelse(property_type %in% c("Townhouse", "Condominium"), "condo", 
                                                  ifelse(property_type %in% c("Bungalow", "House"), "house", "other")))),
         property_category = as.factor(property_category),
         room_type = as.factor(room_type),
         accommodates = ifelse(is.na(accommodates),ifelse(bedrooms == 0, 1, bedrooms),accommodates),
         bedrooms = ifelse(is.na(bedrooms),ceiling(accommodates/2),bedrooms),
         beds = ifelse(is.na(beds),bedrooms,beds),
         bathrooms = ifelse(is.na(bathrooms),median(bathrooms,na.rm = TRUE),bathrooms),
         bed_category = factor(ifelse(bed_type == 'Real Bed', 'bed', 'other')),
         cancellation_policy = factor(ifelse(cancellation_policy %in%  c('super_strict_30','super_strict_60'),'super_strict',cancellation_policy))
  ) %>%
  group_by(state,city,property_category,room_type) %>%
  mutate(avg_square_feet = mean(square_feet,na.rm=TRUE),
         avg_price = mean(price,na.rm=TRUE))%>%
  ungroup() %>%
  mutate(
    square_feet = ifelse(is.na(square_feet),avg_square_feet,square_feet),
    square_feet = ifelse(is.na(square_feet),median(square_feet,na.rm= TRUE),square_feet),
    price = ifelse(is.na(price),avg_price,price),
    price = ifelse(is.na(price),median(price,na.rm=TRUE),price),
    weekly_price = ifelse(is.na(weekly_price),7*price,weekly_price),
    monthly_price = ifelse(is.na(monthly_price),31*price,monthly_price),
    security_deposit = ifelse(is.na(security_deposit),0,security_deposit),
    cleaning_fee = ifelse(is.na(cleaning_fee),0,cleaning_fee)) %>%
  select(-city,-state,-host_acceptance_rate,-host_response_rate,-property_type,-bed_type,-avg_square_feet,-avg_price,-amenities) %>%
  mutate(
    price_per_person = price/accommodates,
    available_30 = as.factor(cut(availability_30,breaks = c(-Inf, 10, 20, 30),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_60 = as.factor(cut(availability_60,breaks = c(-Inf, 20, 40, 60),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_90 = as.factor(cut(availability_90,breaks = c(-Inf, 30, 60, 90),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    available_365 = as.factor(cut(availability_365,breaks = c(-Inf, 122, 243, 365),labels = c("Low", "Medium", "High"),include.lowest = TRUE)),
    features_list = strsplit(features, ","),
    price_per_person = ifelse(is.na(price),0,price/accommodates)) %>%
  group_by(property_category)%>%
  mutate(
    median_ppp = median(price_per_person, na.rm = TRUE)) %>%
  ungroup()%>%
  mutate(
    ppp_ind = ifelse(price_per_person > median_ppp, 1, 0),
    ppp_ind = as.factor(ppp_ind)
  )%>%
  mutate(
    charges_for_extra = as.factor(ifelse(is.na(extra_people) | extra_people == 0,0,1)),
    has_min_nights = as.factor(ifelse(minimum_nights > 1, 1,0)),
    has_cleaning_fee = as.factor(ifelse(as.numeric(cleaning_fee) ==0 | is.na(cleaning_fee) ,0,1)),
    has_security_deposit = as.factor(ifelse(as.numeric(security_deposit) ==0 |is.na(security_deposit) ,0,1))
  )


unique_features <- unique(unlist(test_1$features_list))


for (feature in unique_features) {
  test_1 <- test_1 %>%
    mutate(!!feature := as.integer(sapply(features_list, function(x) feature %in% x)))
}

test_numeric <- test_1 %>%
  select(-features_list,-features,-'NA',-host_since) 


#creating dummies for factor variables on test set
test_dummies <- model.matrix(~ . - 1, data = test_numeric)
test_numeric_final <- data.frame(test_dummies)

##AMENITIES

#preprocessing

train_a <- train_x %>%
  select(id,amenities)%>%
  mutate(amenities = ifelse(is.na(amenities), 'MISSING', amenities))

test_a <- test_x %>%
  select(id,amenities)%>%
  mutate(amenities = ifelse(is.na(amenities),'MISSING',amenities))


cleaning_tokenizer <- function(v) {
  v %>%
    space_tokenizer(sep = ',') 
}

#tokenize
it_train_a <- itoken(train_a$amenities, 
                     preprocessor = tolower,
                     tokenizer = cleaning_tokenizer, 
                     ids = train_a$id, 
                     progressbar = FALSE)

it_test_a <- itoken(test_a$amenities, 
                    preprocessor = tolower,
                    tokenizer = cleaning_tokenizer, 
                    ids = test_a$id, 
                    progressbar = FALSE)


vocab_a <- create_vocabulary(it_train_a)

#vectorize
vectorizer_a <- vocab_vectorizer(vocab_a)
dtm_train_a <- create_dtm(it_train_a, vectorizer_a)
dtm_test_a <- create_dtm(it_test_a, vectorizer_a)


amenities_train <- data.frame(as.matrix(dtm_train_a))
amenities_test <- data.frame(as.matrix(dtm_test_a))

#lasso regression

set.seed(1)

idx_a <- sample(nrow(amenities_train),.7*nrow(amenities_train))

amenities_y <- train_y %>%
  select(perfect_rating_score)

x_train <- as.matrix(amenities_train[idx_a, ])
x_valid <- as.matrix(amenities_train[-idx_a, ])


y_train <- amenities_y[idx_a, , drop = TRUE]  
y_valid <- amenities_y[-idx_a, , drop = TRUE]


#model

lambda_values <- 10^seq(-4, 1, length = 100)

#Performing cross-validation for lambda selection
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = lambda_values)

#Plotting the mean cross-validated error for each lambda
plot(cv_fit)


best_lambda <- cv_fit$lambda.min
print(paste("Best lambda:", best_lambda))

#Refitting the model using the best lambda

final_model <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = best_lambda)

#selecting 20 most significant features
coefficients_a <- coef(final_model, s = "lambda.min")[-1, , drop = FALSE] 
non_zero_coefs <- coefficients_a[coefficients_a[, 1] != 0, , drop = FALSE]
sorted_features <- sort(abs(non_zero_coefs[, 1]), decreasing = TRUE)
top_20_features <- names(sorted_features)[1:20]
print(top_20_features)


#getting only those 20 features

train_a <- amenities_train[, c(top_20_features)]
test_a <-  amenities_test[, c(top_20_features)]

#converting into factors

train_a[] <- lapply(train_a, function(x) factor(x, levels = c(0, 1)))
test_a[] <- lapply(test_a, function(x) factor(x, levels = c(0, 1)))

train_a <- train_a %>%
  mutate(across(where(is.factor), ~ fct_na_value_to_level(., "0")))

test_a <- test_a %>%
  mutate(across(where(is.factor), ~ fct_na_value_to_level(., "0")))


##ZIPCODES

#preprocessing

train_z <- train_x %>%
  select(zipcode)%>%
  mutate(zipcode = ifelse(is.na(zipcode),'MISSING',as.character(zipcode)))

test_z <- test_x %>%
  select(zipcode)%>%
  mutate(zipcode = ifelse(is.na(zipcode),'MISSING',as.character(zipcode)))


#fetching unique zips in both train and test datasets and combining them

unique_zips_train <- unique(train_z$zipcode)
unique_zips_test <- unique(test_z$zipcode)
unique_zips_all <- union(unique_zips_train, unique_zips_test)


# Creating dummy variables
add_zipcode_dummies <- function(data, zipcode_levels) {
  
  data$zipcode <- factor(data$zipcode, levels = zipcode_levels)
  zipcode_dummies <- model.matrix(~ zipcode - 1, data = data)
  zipcode_dummies <- as.data.frame(zipcode_dummies)
  
  #Remove the original zipcode column
  data <- select(data, -zipcode)
  
  # Return the modified dataset with dummy variables added
  return(cbind(data, zipcode_dummies))
}

#Applying the function to train and test datasets
train_z <- add_zipcode_dummies(train_z, unique_zips_all)
test_z <- add_zipcode_dummies(test_z, unique_zips_all)


set.seed(1)

idx <- sample(nrow(train_z),.7*nrow(train_z))

zip_y <- train_y %>%
  select(perfect_rating_score)

zip_x <-train_z %>%
  select(starts_with('zipcode'))

x_train <- as.matrix(zip_x[idx, ])
x_valid <- as.matrix(zip_x[-idx, ])


y_train <- zip_y[idx, , drop = TRUE] 
y_valid <- zip_y[-idx, , drop = TRUE]


#model

lambda_values <- 10^seq(-4, 1, length = 100)

# Performing cross-validation for lambda selection
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = lambda_values)

# Plot of the the mean cross-validated error for each lambda
plot(cv_fit)


best_lambda <- cv_fit$lambda.min
print(paste("Best lambda:", best_lambda))


#Refitting  the model using the best lambda
final_model <- glmnet(x_train, y_train, alpha = 1, family = "binomial", lambda = best_lambda)

#selecting 20 most significant features
coefficients_z <- coef(final_model, s = "lambda.min")[-1, , drop = FALSE] 
non_zero_coefs <- coefficients_z[coefficients_z[, 1] != 0, , drop = FALSE]
sorted_features <- sort(abs(non_zero_coefs[, 1]), decreasing = TRUE)
top_20_features <- names(sorted_features)[1:20]
print(top_20_features)


#selecting only significant zips 
train_z <- train_z %>%
  select(all_of(top_20_features))

#Selecting only the significant zipcode columns from the test dataset
test_z <- test_z %>%
  select(all_of(top_20_features))

train_z[] <- lapply(train_z, function(x) factor(x, levels = c(0, 1)))
test_z[] <- lapply(test_z, function(x) factor(x, levels = c(0, 1)))

##Description

train_d <- train_x%>%
  select(description,id)

test_d <- test_x%>%
  select(description,id)


cleaning_tokenizer_d <- function(v) {
  v %>%
    removePunctuation %>% 
    removeWords(tm::stopwords(kind="en")) %>% 
    stemDocument %>%
    word_tokenizer 
}

it_train_d <- itoken(train_d$description, 
                     preprocessor = tolower,
                     tokenizer = cleaning_tokenizer_d, 
                     ids = train_d$id, 
                     progressbar = FALSE)


it_test_d <- itoken(test_d$description, 
                    preprocessor = tolower, 
                    tokenizer = cleaning_tokenizer_d, 
                    ids = test_d$id, 
                    progressbar = FALSE)

vocab_d <- create_vocabulary(it_train_d, ngram = c(1L, 2L))

vocab_final_d <- prune_vocabulary(vocab_d, term_count_min = 100, doc_proportion_max = 0.5)
vectorizer_d <- vocab_vectorizer(vocab_final_d)

# Convert the training documents into a DTM
dtm_train_d <- create_dtm(it_train_d, vectorizer_d)
dtm_test_d <-  create_dtm(it_test_d, vectorizer_d)

# tf-idf
tfidf_model <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train_d, tfidf_model)
dtm_test_tfidf <- fit_transform(dtm_test_d, tfidf_model)

train_rows <- sample(nrow(train_d), 0.7*nrow(train_d))
tr_dtm_d <- dtm_train_tfidf [train_rows,]
va_dtm_d <- dtm_train_tfidf [-train_rows,]

train_y_vector <- train_y[[1]]

# Get the y values
tr_y_d <- train_y_vector[train_rows]
va_y_d <- train_y_vector[-train_rows]



lambda <- 10^seq(-7,7, length = 10)

acc <- rep(0,length(lambda))
for (i in c(1:length(lambda))){
  lasso_model <- glmnet(tr_dtm_d, tr_y_d, family="binomial", alpha=1, lambda=lambda[i])
  prediction_lasso <- predict(lasso_model, newx = va_dtm_d, type="response")
  class_lasso <- ifelse(prediction_lasso > 0.5,1,0)
  acc_lasso <- mean(ifelse(class_lasso == va_y_d, 1, 0))
  acc[i] <- acc_lasso}

best_lambda_lasso <- lambda[which.max(acc)]
best_accuracy_lasso <- acc[which.max(acc)]

best_lambda_lasso
best_accuracy_lasso


final_lasso_model <- glmnet(tr_dtm_d, tr_y_d, family = "binomial", alpha = 1, lambda = best_lambda_lasso)
coefficients_des <- coef(final_lasso_model, s = "lambda.min")[-1, , drop = FALSE]  
non_zero_coefs <- coefficients_des[coefficients_des[, 1] != 0, , drop = FALSE]

# Sort features by the absolute value of their coefficients

sorted_features <- sort(abs(non_zero_coefs[, 1]), decreasing = TRUE)
top_20_features <- names(sorted_features)[1:20]
print(top_20_features)


train_des  <- dtm_train_tfidf[,top_20_features]
test_des <- dtm_test_tfidf[,top_20_features]

train_des <- as.data.frame(as.matrix(train_des))
test_des <- as.data.frame(as.matrix(test_des))

##combining all the dataframes together

train_data_final <- cbind(train_numeric_final,train_a,train_z,train_des)
test_data_final <- cbind(test_numeric_final,test_a,test_z,test_des)

colnames(train_data_final) <- gsub(" ", "_", colnames(train_data_final))
colnames(test_data_final) <- gsub(" ", "_", colnames(test_data_final))

#removing duplicate columns
remove_duplicate_columns <- function(df) {
  dup_cols <- names(df)[duplicated(names(df))]
  keep_cols <- !duplicated(names(df)) | !names(df) %in% dup_cols
  df <- df[, keep_cols]
  return(df)
}

train_data_final <- remove_duplicate_columns(train_data_final)
test_data_final <- remove_duplicate_columns(test_data_final)

#dealing with missing columns in test
missing_cols <- setdiff(names(train_data_final), names(test_data_final))
missing_cols

#Adding missing columns with 0
for (col in missing_cols) {
  test_data_final[[col]] <- 0 
}

test_data_final <- test_data_final[names(train_data_final)]

###EDA

#Plot for host response time
train_eda <- cbind(train_x,train_y)
train_eda <- train_eda %>%
  filter(!is.na(host_response_time))



ggplot(train_eda, aes(x = host_response_time, fill =perfect_rating_score)) +
  geom_bar(position = "stack") +
  ggtitle("Host Response Time vs. Perfect Rating Scores") +
  xlab("Host Response Time") +
  ylab("Count") +
  scale_fill_manual(values = c("1" = "cornflowerblue", "0" = "tomato"), name = "Perfect Rating") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

##Plot for property type
ggplot(train_1, aes(x = property_category)) +
  geom_bar(fill = "coral") +
  ggtitle("Distribution of Property Type") +
  xlab("Property Type") +
  ylab("Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Plot for price distribution by property category without outliers
ggplot(train_1, aes(x = property_category, y = price)) +
  geom_boxplot(fill = "lightblue", color = "darkblue", outlier.shape = NA) +
  ggtitle("Price Distribution by Property Category") +
  xlab("Property Category") +
  ylab("Price") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


#Bar chart between room type and average availability_30

average_availability_data <- train_1 %>%
  group_by(room_type) %>%
  summarize(average_availability_30 = mean(availability_30, na.rm = TRUE))



ggplot(average_availability_data, aes(x = room_type, y = average_availability_30, fill = room_type)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  ggtitle("Bar Chart of Room Type vs. Average Availability (30 Days)") +
  xlab("Room Type") +
  ylab("Average Availability in Next 30 Days") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title = element_text(size = 12),
    legend.position = "none") +
  scale_fill_brewer(palette = "Set3")

#Bar plot for top 5 cities with most listings

top_cities_data <- train_x %>%
  count(smart_location, sort = TRUE) %>%
  top_n(5, wt = n)

ggplot(top_cities_data, aes(x = reorder(smart_location, -n), y = n, fill = smart_location)) +
  geom_bar(stat = "identity", color = "black", width = 0.7) +
  ggtitle("Top 5 Cities with Most Listings") +
  xlab("City") +
  ylab("Number of Listings") +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
    axis.text.y = element_text(size = 10),
    axis.title = element_text(size = 12),
    legend.position = "none" # Hide legend since bars are labeled
  ) +
  scale_fill_brewer(palette = "Set3")


##CLASSIFICATION MODELS

set.seed(10)
folds <- createFolds(train_y$perfect_rating_score, k = 5, list = TRUE, returnTrain = FALSE)
results <- data.frame(Model = character(), TPR = numeric(), FPR = numeric())

evaluate_model <- function(predictions, y_valid) {
  predictions <- factor(predictions, levels = c(0, 1))
  y_valid <- factor(y_valid, levels = c(0, 1))
  cm <- table(Predicted = predictions, Actual = y_valid)
  tpr <- sensitivity(cm, positive = "1")
  fpr <- 1 - specificity(cm, negative = "0")
  list(tpr = tpr, fpr = fpr)
}

#Model - 1 : Lasso Regression

#parameter tuning

lambda_values <- 10^seq(-4, 1, length = 100)
cv_fit_lasso <- cv.glmnet(as.matrix(train_data_final), train_y$perfect_rating_score, alpha = 1, family = "binomial", nfolds = 5, lambda = lambda_values)
par(mar = c(5, 4, 4, 2) + 0.3, xpd = NA)
plot(cv_fit_lasso)
title(main = "Lasso Regression cross validation results", line = 2)

best_lambda_lasso <- cv_fit_lasso$lambda.min
cat("Best Lambda for Lasso: ", best_lambda_lasso, "\n")

#Training Performance

model_full <- glmnet(as.matrix(train_data_final), train_y$perfect_rating_score, alpha = 1, family = "binomial", lambda = best_lambda_lasso)
training_predictions <- predict(model_full, newx = as.matrix(train_data_final), type = "response")
training_class_predictions <- ifelse(training_predictions > 0.5, 1, 0)
training_eval <- evaluate_model(training_class_predictions, train_y$perfect_rating_score)

cat("Training TPR for Lasso: ", training_eval$tpr, "\n")
cat("Training FPR for Lasso: ", training_eval$fpr, "\n")

# Initialize vectors to store TPR and FPR for each fold
fold_tpr <- numeric(length(folds))
fold_fpr <- numeric(length(folds))


#Evaluation

for (fold in seq_along(folds)) {
  train_indices <- setdiff(seq_len(nrow(train_data_final)), folds[[fold]])
  test_indices <- folds[[fold]]
  
  x_train <- train_data_final[train_indices, ]
  y_train <- train_y$perfect_rating_score[train_indices]
  x_test <- train_data_final[test_indices, ]
  y_test <- train_y$perfect_rating_score[test_indices]
  
  model <- glmnet(as.matrix(x_train), y_train, alpha = 1, family = "binomial", lambda = best_lambda_lasso)
  predictions <- predict(model, newx = as.matrix(x_test), type = "response")
  class_predictions <- ifelse(predictions > 0.5, 1, 0)
  eval <- evaluate_model(class_predictions, y_test)
  
  fold_tpr[fold] <- eval$tpr
  fold_fpr[fold] <- eval$fpr
  results <- rbind(results, data.frame(Model = "Lasso Regression", Fold = fold, TPR = eval$tpr, FPR = eval$fpr))
}

cat("Generalization TPR of Lasso Regression : ", mean(fold_tpr), "\n")
cat("Generalization  FPR of Lasso Regression : ", mean(fold_fpr), "\n")


# Model 2 : Ridge_Regression 

#parameter tuning

lambda_values_ridge <- 10^seq(-4, 1, length = 100)
cv_fit_ridge <- cv.glmnet(as.matrix(train_data_final), train_y$perfect_rating_score, alpha = 0, family = "binomial", nfolds = 5, lambda = lambda_values_ridge)
par(mar = c(5, 4, 4, 2) + 0.3, xpd = NA) 
plot(cv_fit_ridge)
title(main = "Ridge Regression Cross-Validation Results", line = 2)

best_lambda_ridge <- cv_fit_ridge$lambda.min
cat("Best Lambda for Ridge: ", best_lambda_ridge, "\n")

#Training Performance
model_full <- glmnet(as.matrix(train_data_final), train_y$perfect_rating_score, alpha = 0, family = "binomial", lambda = best_lambda_ridge)
training_predictions <- predict(model_full, newx = as.matrix(train_data_final), type = "response")
training_class_predictions <- ifelse(training_predictions > 0.5, 1, 0)
training_eval <- evaluate_model(training_class_predictions, train_y$perfect_rating_score)

cat("Training TPR for Ridge: ", training_eval$tpr, "\n")
cat("Training FPR for Ridge: ", training_eval$fpr, "\n")


#Evaluation
fold_tpr <- numeric(length(folds))
fold_fpr <- numeric(length(folds))

for (fold in seq_along(folds)) {
  train_indices <- setdiff(seq_len(nrow(train_data_final)), folds[[fold]])
  test_indices <- folds[[fold]]
  
  x_train <- train_data_final[train_indices, ]
  y_train <- train_y$perfect_rating_score[train_indices]
  x_test <- train_data_final[test_indices, ]
  y_test <- train_y$perfect_rating_score[test_indices]
  
  model <- glmnet(as.matrix(x_train), y_train, alpha = 0, family = "binomial", lambda = best_lambda_ridge)
  predictions <- predict(model, newx = as.matrix(x_test), type = "response")
  class_predictions <- ifelse(predictions > 0.5, 1, 0)
  eval <- evaluate_model(class_predictions, y_test)
  
  fold_tpr[fold] <- eval$tpr
  fold_fpr[fold] <- eval$fpr
  results <- rbind(results, data.frame(Model = "Ridge Regression", Fold = fold, TPR = eval$tpr, FPR = eval$fpr))
}

cat("Generalization TPR of Ridge Regression : ", mean(fold_tpr), "\n")
cat("Generalization FPR of Ridge Regression : ", mean(fold_fpr), "\n")

#Model 3 : DECISION-TREE

raw_data_final <- cbind(train_data_final,train_y)
set.seed(20)
valid_instn <- sample(nrow(raw_data_final), 0.30*nrow(raw_data_final))
data_valid <- raw_data_final[valid_instn,]
data_train <- raw_data_final[-valid_instn,]


#parameter tuning

max_depth <- seq(5,20, by = 5)
performance_dt <- data.frame(max_depth =max_depth, tpr = rep(0, length(max_depth)), fpr = rep(0, length(max_depth)))

mycontrol <- tree.control(nobs = nrow(data_train), mincut = 1, minsize = 2, mindev = 0.0005)
model <- tree(perfect_rating_score ~ ., data = data_train , control = mycontrol)


for (i in seq_along(max_depth)) {
  
  pruned_tree <- prune.tree(model, best = max_depth[i])
  pred <- predict(pruned_tree, newdata = data_valid)
  probabilities <- pred[,2]
  classifications <- ifelse(probabilities > 0.5,1,0)
  eval <- evaluate_model(classifications, data_valid$perfect_rating_score)
  
  performance_dt[i, "tpr"] <- eval$tpr
  performance_dt[i, "fpr"] <- eval$fpr
  
}

best_model_dt <- performance_dt[performance_dt$fpr < 0.1 & performance_dt$tpr == max(performance_dt$tpr), ]
best_max_depth <- max(best_model_dt$max_depth)

ggplot(performance_dt, aes(x = max_depth)) +
  geom_line(aes(y = tpr, color = "TPR")) +
  geom_line(aes(y = fpr, color = "FPR")) +
  geom_point(aes(y = tpr, color = "TPR")) +
  geom_point(aes(y = fpr, color = "FPR")) +
  scale_color_manual(values = c("TPR" = "cornflowerblue", "FPR" = "tomato")) +
  ggtitle("Decision Tree Performance") +
  xlab("Max Depth of the Pruned tree") +
  ylab("Rate") +
  theme_minimal()

cat("Best max depth for Decision Tree: ", best_max_depth, "\n")

#Estimating training performance

mycontrol <- tree.control(nobs = nrow(data_train), mincut = 1, minsize = 2, mindev = 0.0005)
model <- tree(perfect_rating_score ~ ., data = data_train, control = mycontrol)
pruned_tree <- prune.tree(model, best = best_max_depth)
predictions_train <- predict(pruned_tree, newdata = data_train)
probabilities <- predictions_train[,2]
classifications_train <- ifelse(probabilities > 0.5,1,0)
train_eval <- evaluate_model(classifications_train, data_train$perfect_rating_score)

cat("Training TPR for Decision Tree: ", train_eval$tpr, "\n")
cat("Training FPR for Decision Tree: ", train_eval$fpr, "\n")


#Evaluation
fold_tpr <- numeric(length(folds))
fold_fpr <- numeric(length(folds))

for (fold in seq_along(folds)) {
  train_indices <- setdiff(seq_len(nrow(train_data_final)), folds[[fold]])
  test_indices <- folds[[fold]]
  
  x_train <- train_data_final[train_indices, ]
  y_train <- train_y$perfect_rating_score[train_indices]
  x_test <- train_data_final[test_indices, ]
  y_test <- train_y$perfect_rating_score[test_indices]
  
  mycontrol <- tree.control(nobs = nrow(x_train), mincut = 1, minsize = 2, mindev = 0.0005)
  model <- tree(perfect_rating_score ~ ., data = data.frame(x_train, perfect_rating_score = y_train), control = mycontrol)
  pruned_tree <- prune.tree(model, best = best_max_depth)
  predictions <- predict(pruned_tree, newdata = data.frame(x_test))
  probabilities <- predictions[,2]
  classifications <- ifelse(probabilities > 0.5,1,0)
  eval <- evaluate_model(classifications, y_test)
 
  
  fold_tpr[fold] <- eval$tpr
  fold_fpr[fold] <- eval$fpr
  results <- rbind(results, data.frame(Model = "Decision Tree", Fold = fold, TPR = eval$tpr, FPR = eval$fpr))
}

cat("Generalization TPR of Decision Tree : ", mean(fold_tpr), "\n")
cat("Generalization FPR of Decision Tree : ", mean(fold_fpr), "\n")

#MODEL 4 : RANDOM-FOREST

#Parameter tuning
ntrees <- seq(100, 600, by = 100)
performance_rf <- data.frame(ntrees = ntrees, tpr = rep(0, length(ntrees)), fpr = rep(0, length(ntrees)))

for (i in 1:length(ntrees)) {
  model <- randomForest(perfect_rating_score ~ ., data = data_train,ntree = ntrees[i], mtry = 11)
  predictions <- predict(model, newdata =data_valid , type = "prob")
  class_predictions <- ifelse(predictions[,2] > 0.5, 1, 0)
  eval <- evaluate_model(class_predictions,data_valid$perfect_rating_score)
  performance_rf[i, "tpr"] <- eval$tpr
  performance_rf[i, "fpr"] <- eval$fpr
}

# Select the best model based on TPR and FPR
best_model <- performance_rf[performance_rf$fpr < 0.1 & performance_rf$tpr == max(performance_rf$tpr), ]
best_ntree <- min(best_model$ntrees)

# Plot performance
ggplot(performance_rf, aes(x = ntrees)) +
  geom_line(aes(y = tpr, color = "TPR")) +
  geom_line(aes(y = fpr, color = "FPR")) +
  geom_point(aes(y = tpr, color = "TPR")) +
  geom_point(aes(y = fpr, color = "FPR")) +
  scale_color_manual(values = c("TPR" = "cornflowerblue", "FPR" = "tomato")) +
  ggtitle("Random Forest Performance") +
  xlab("Number of Trees") +
  ylab("Rate") +
  theme_minimal()

cat("Best nunmer of trees for Random Forest: ", best_ntree, "\n")

#Evaluating Training Performance
final_model <- randomForest(perfect_rating_score ~ ., data = raw_data_final, ntree = best_ntree, mtry = 11)
training_predictions <- predict(final_model, newdata = raw_data_final, type = "prob")
training_class_predictions <- ifelse(training_predictions[,2] > 0.5, 1, 0)
training_eval <- evaluate_model(training_class_predictions, raw_data_final$perfect_rating_score)


cat("Training TPR for Random Forest: ", training_eval$tpr, "\n")
cat("Training FPR for Random Forest: ", training_eval$fpr, "\n")

#Evaluation
fold_tpr <- numeric(length(folds))
fold_fpr <- numeric(length(folds))

for (fold in seq_along(folds)) {
  train_indices <- setdiff(seq_len(nrow(train_data_final)), folds[[fold]])
  test_indices <- folds[[fold]]
  
  x_train <- train_data_final[train_indices, ]
  y_train <- train_y$perfect_rating_score[train_indices]
  x_test <- train_data_final[test_indices, ]
  y_test <- train_y$perfect_rating_score[test_indices]
  
  train_data <- data.frame(x_train, perfect_rating_score = y_train)
  model <- randomForest(perfect_rating_score ~ ., data = train_data, ntree = best_ntree, mtry = 11)
  predictions <- predict(model, newdata =x_test , type = "prob")
  class_predictions <- ifelse(predictions[,2] > 0.5, 1, 0)
  eval <- evaluate_model(class_predictions,y_test)

  fold_tpr[fold] <- eval$tpr
  fold_fpr[fold] <- eval$fpr
  results <- rbind(results, data.frame(Model = "Random Forest", Fold = fold, TPR = eval$tpr, FPR = eval$fpr))
}

cat("Generalization TPR of Random Forest : ", mean(fold_tpr), "\n")
cat("Generalization FPR of Random Forest : ", mean(fold_fpr), "\n")

## MODEL 5 : XGBOOST

etas <- seq(0.01, 0.1, by = 0.01)

performance_xg <- data.frame(etas = etas, tpr = rep(0, length(etas)), fpr = rep(0, length(etas)))

x_train <- data_train %>%
               select(-perfect_rating_score)  %>% 
               mutate(across(where(is.factor),as.numeric))

x_valid <- data_valid %>%
  select(-perfect_rating_score)  %>% 
  mutate(across(where(is.factor),as.numeric))

y_train <- as.numeric(as.character(data_train$perfect_rating_score))
y_valid <- as.numeric(as.character(data_valid$perfect_rating_score))

for (i in seq_along(etas)) {
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = etas[i],
    max_depth = 5,
    min_child_weight = 5,
    subsample = 0.6,
    colsample_bytree = 0.5
  )
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  dvalid <- xgb.DMatrix(data = as.matrix(x_valid), label = y_valid)
  
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
  predictions <- predict(xgb_model, dvalid)
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  eval <- evaluate_model(predicted_classes, y_valid)
  
  performance_xg[i, "tpr"] <- eval$tpr
  performance_xg[i, "fpr"] <- eval$fpr
}

best_model <- performance_xg[performance_xg$fpr < 0.1 & performance_xg$tpr == max(performance_xg$tpr), ]
best_eta <- max(best_model$etas)

# Plot performance

ggplot(performance_xg, aes(x = etas)) +
  geom_line(aes(y = tpr, color = "TPR")) +
  geom_line(aes(y = fpr, color = "FPR")) +
  geom_point(aes(y = tpr, color = "TPR")) +
  geom_point(aes(y = fpr, color = "FPR")) +
  scale_color_manual(values = c("TPR" = "cornflowerblue", "FPR" = "tomato")) +
  ggtitle("XGBoost Performance") +
  xlab("Learning Rate") +
  ylab("Rate") +
  theme_minimal()

cat("Best learning rate  for XGBoost: ", best_eta, "\n")

# Training Performance
params$eta <- best_eta
dtrain_full <- xgb.DMatrix(data = as.matrix(train_data_final %>% mutate(across(where(is.factor), as.numeric))), 
                           label = as.numeric(as.character(train_y$perfect_rating_score)))
final_model <- xgb.train(params = params, data = dtrain_full, nrounds = 500,verbose = 0)
training_predictions <- predict(final_model, dtrain_full)
training_class_predictions <- ifelse(training_predictions > 0.5, 1, 0)
training_eval <- evaluate_model(training_class_predictions, train_y$perfect_rating_score)

cat("Training TPR for XGBoost: ", training_eval$tpr, "\n")
cat("Training FPR for XGBoost: ", training_eval$fpr, "\n")

# XGBoost Evaluation

fold_tpr <- numeric(length(folds))
fold_fpr <- numeric(length(folds))

for (fold in seq_along(folds)) {
  train_indices <- setdiff(seq_len(nrow(train_data_final)), folds[[fold]])
  test_indices <- folds[[fold]]
  
  x_train <- train_data_final[train_indices, ] %>% mutate(across(where(is.factor), as.numeric))
  y_train <- as.numeric(as.character(train_y$perfect_rating_score[train_indices]))
  x_test <- train_data_final[test_indices, ] %>% mutate(across(where(is.factor), as.numeric))
  y_test <- as.numeric(as.character(train_y$perfect_rating_score[test_indices]))
  
  dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
  dtest <- xgb.DMatrix(data = as.matrix(x_test), label = y_test)
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "logloss",
    eta = best_eta,
    max_depth = 5,
    min_child_weight = 5,
    subsample = 0.6,
    colsample_bytree = 0.5
  )
  
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
  predictions <- predict(xgb_model, dtest)
  predicted_classes <- ifelse(predictions > 0.5, 1, 0)
  eval <- evaluate_model(predicted_classes, y_test)
  
  fold_tpr[fold] <- eval$tpr
  fold_fpr[fold] <- eval$fpr
  results <- rbind(results, data.frame(Model = "XGBoost", Fold = fold, TPR = eval$tpr, FPR = eval$fpr))
}

cat("Generalization TPR of XGBoost : ", mean(fold_tpr), "\n")
cat("Generalization FPR of XGBoost : ", mean(fold_fpr), "\n")


#XGBOOST HAS THE BEST GENERALIZATION PERFORMANCE SO FAR
## PLOTTING TPR AND FPR FOR DIFFERENT CUTOFF VALUES TO SELECT ONE CUTOFF
#SIMPLE train and validation split is used for this 

set.seed(3)

train_instn <- sample(nrow(train_data_final), 0.7*nrow(train_data_final))

x_train <- train_data_final[train_instn, ] %>% mutate(across(where(is.factor), as.numeric))
y_train <- as.numeric(as.character(train_y$perfect_rating_score[train_instn]))

x_test <- train_data_final[-train_instn, ] %>% mutate(across(where(is.factor), as.numeric))
y_test <- as.numeric(as.character(train_y$perfect_rating_score[-train_instn]))

dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = y_train)
dvalid <- xgb.DMatrix(data = as.matrix(x_test), label = y_test)

new_params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 5,
  min_child_weight = 5,
  subsample = 0.6,
  colsample_bytree = 0.5
)

xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500, verbose = 0)
predictions <- predict(xgb_model, dvalid)

cutoffs <- seq(0.3, 0.7, by = 0.05)
tpr_values <- numeric(length(cutoffs))
fpr_values <- numeric(length(cutoffs))

for (i in seq_along(cutoffs)) {
  cutoff <- cutoffs[i]
  predicted_classes <- ifelse(predictions > cutoff, 1, 0)
  eval <- evaluate_model(predicted_classes, y_test)
  tpr_values[i] <- eval$tpr
  fpr_values[i] <- eval$fpr
}


performance_df <- data.frame(
  cutoff = cutoffs,
  tpr = tpr_values,
  fpr = fpr_values
)

performance_df


# Plot the TPR and FPR against Cutoff values
ggplot(performance_df, aes(x = Cutoff)) +
  geom_line(aes(y = TPR, color = "TPR")) +
  geom_line(aes(y = FPR, color = "FPR")) +
  geom_point(aes(y = TPR, color = "TPR")) +
  geom_point(aes(y = FPR, color = "FPR")) +
  scale_color_manual(values = c("TPR" = "blue", "FPR" = "red")) +
  ggtitle("TPR and FPR vs Cutoff Values for XGBoost") +
  xlab("Cutoff Value") +
  ylab("Rate") +
  theme_minimal()


#Code to generate predictions for Submission-----------------------------------------------
library(xgboost)

#converting factors to numeric
train_data_final[] <- lapply(train_data_final, function(x) {
  if (is.factor(x)) as.numeric(x) else x
})

test_data_final[] <- lapply(test_data_final, function(x) {
  if (is.factor(x)) as.numeric(x) else x
})


train_y_numeric <- as.numeric(as.character(train_y$perfect_rating_score)) 



dtrain <- xgb.DMatrix(data = as.matrix(train_data_final), label = train_y_numeric, feature_name = colnames(train_data_final))
dtest <- xgb.DMatrix(data = as.matrix(test_data_final),feature_name = colnames(test_data_final))


params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "logloss",
  eta = 0.1,
  max_depth = 5,
  min_child_weight = 5,
  subsample = 0.6,
  colsample_bytree = 0.5
)


#Train the model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  verbose = 0  
)


test_probs <- predict(xgb_model, dtest)
predicted_classes <- ifelse(test_probs > 0.5, "YES","NO")
classifications_perfect <- ifelse(is.na(predicted_classes), "NO", predicted_classes)
classifications_perfect <- factor(classifications_perfect, levels = c("YES", "NO"))
assertthat::assert_that(sum(is.na(classifications_perfect)) == 0)

# Show table of results
table(predicted_classes)
table(classifications_perfect)

write.table(as.character(classifications_perfect), "perfect_rating_score_group8.csv", row.names = FALSE, quote = FALSE)

