#Import the required libraries
library(ggplot2)
library(gridExtra)
library(cowplot)
library(PCAmixdata)
library(randomForest)
library(psych)
library(MASS)
library(klaR)
library(caTools)
library(gbm)
library(corrplot)

data = read.csv("D:/Fall 2023/MGSC661/Final Project/Dataset 5 - Automobile data.csv")
attach(data)

#Exploratory Data Analysis and Pre-processing

#1. Check for missing values in the data
colSums(data == '?')

#2. Remove columns that are not related to determining fuel efficiency of car
data = subset(data, select = -c(symboling, normalized.losses, price))

#3. Create a new column for average mileage - combining city and highway
data$avg_mpg = (data$city.mpg + data$highway.mpg) / 2
data = data[, !names(data) %in% c("city.mpg", "highway.mpg")]

#4. Function to Convert text numbers to numeric format
convert_to_number <- function(column) {
  unique_values <- unique(column)
  mapping <- setNames(1:length(unique_values), unique_values)
  return(mapping[column])
}

#5. Fill missing values in different fields

#5(i) Replace missing values in num-of-doors with the mode value for the same body-type
data$num.of.doors[data$num.of.doors == '?'] = NA
mode_doors <- aggregate(num.of.doors ~ body.style, data = data, FUN = function(x) {
  tab = table(x)
  names(tab)[which.max(tab)]
})
for (i in 1:nrow(mode_doors)) {
  body.style = mode_doors[i, "body.style"]
  mode_value = mode_doors[i, "num.of.doors"]
  data$num.of.doors[data$body.style == body.style & is.na(data$num.of.doors)] = mode_value
}

data$num.of.doors <- convert_to_number(data$num.of.doors)
data$num.of.cylinders <- convert_to_number(data$num.of.cylinders)

#5(ii) For engine related fields - bore, stroke, horsepower, peak.rpm - Use regression trees to fill missing values

# Select Engine related columns for training the model
engine_data = data.frame(bore = as.numeric(data$bore),
                          stroke = as.numeric(data$stroke),
                          horsepower = as.numeric(data$horsepower),
                          peak.rpm = as.numeric(data$peak.rpm),
                          make = as.factor(data$make),
                          fuel.type = as.factor(data$fuel.type),
                          aspiration = as.factor(data$aspiration),
                          engine.type = as.factor(data$engine.type),
                          num.of.cylinders = as.factor(data$num.of.cylinders),
                          engine.size = data$engine.size,
                          fuel.system = as.factor(data$fuel.system))

bore_train = engine_data[!is.na(engine_data$bore), , drop = FALSE]
bore_train = bore_train[, !names(bore_train) %in% c("stroke", "horsepower", "peak.rpm"), drop = FALSE]
stroke_train = engine_data[!is.na(engine_data$stroke), , drop = FALSE]
stroke_train = stroke_train[, !names(stroke_train) %in% c("bore", "horsepower", "peak.rpm"), drop = FALSE]
horsepower_train = engine_data[!is.na(engine_data$horsepower), , drop = FALSE]
horsepower_train = horsepower_train[, !names(horsepower_train) %in% c("bore", "stroke", "peak.rpm"), drop = FALSE]
peak.rpm_train = engine_data[!is.na(engine_data$peak.rpm), , drop = FALSE]
peak.rpm_train = peak.rpm_train[, !names(peak.rpm_train) %in% c("bore", "stroke", "horsepower"), drop = FALSE]

set.seed(123)
rf_bore = randomForest(bore_train$bore ~ ., data = bore_train, importance = TRUE)
rf_stroke = randomForest(stroke_train$stroke ~ ., data = stroke_train, importance = TRUE)
rf_horsepower = randomForest(horsepower_train$horsepower ~ ., data = horsepower_train, importance = TRUE)
rf_peak.rpm = randomForest(peak.rpm_train$peak.rpm ~ ., data = peak.rpm_train, importance = TRUE)

bore_predict = data.frame(engine_data[is.na(engine_data$bore),,])
bore_predict$bore = round(predict(rf_bore, newdata = bore_predict),2)
stroke_predict = data.frame(engine_data[is.na(engine_data$stroke),,])
stroke_predict$stroke = round(predict(rf_stroke, newdata = stroke_predict),2)
horsepower_predict = data.frame(engine_data[is.na(engine_data$horsepower),,])
horsepower_predict$horsepower = round(predict(rf_horsepower, newdata = horsepower_predict),2)
peak.rpm_predict = data.frame(engine_data[is.na(engine_data$peak.rpm),,])
peak.rpm_predict$peak.rpm = round(predict(rf_peak.rpm, newdata = peak.rpm_predict),2)

data$bore[match(rownames(bore_predict), rownames(data))] <- bore_predict$bore
data$stroke[match(rownames(stroke_predict), rownames(data))] <- stroke_predict$stroke
data$horsepower[match(rownames(horsepower_predict), rownames(data))] <- horsepower_predict$horsepower
data$peak.rpm[match(rownames(peak.rpm_predict), rownames(data))] <- peak.rpm_predict$peak.rpm


#6. Convert data types in the final data

num_fields = c(4,8,9,10,11,12,14,15,17,18,19,20,21,22)
cat_fields = c(1,2,3,5,6,7,13,16)

data[, num_fields] <- lapply(data[, num_fields], as.numeric)
data[, cat_fields] <- lapply(data[, cat_fields], as.factor)


##########START CODE FOR CLUSTERING MODEL##########

#1. Principal Component Analysis for keeping 2 dimensions for clustering

numeric_data = data[, num_fields]
scaled_numeric_data = as.data.frame(lapply(numeric_data, scale))
scaled_data = data
scaled_data[, num_fields] = scaled_numeric_data

X.num = scaled_data[,num_fields]
X.cat = scaled_data[,cat_fields]

pca = PCAmix(X.num,X.cat,ndim=2,graph=TRUE)
coords = as.data.frame(pca$ind$coord)

#2. K-means clustering using PCA coordinates in 2 dimensions
#Using cluster size=4 based on the PCA plot
km = kmeans(coords, centers = 4)
plot(coords[,1], coords[,2], col = km$cluster, pch = 20, xlab = "Principal Component 1", ylab = "Principal Component 2", main="Clustering over Principal Components")
data$cluster<-as.factor(km$cluster)

#3. Visualize the distribution of features across clusters

p1 = ggplot(data, aes(cluster, num.of.cylinders, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Number of Cylinders")

p2 = ggplot(data, aes(cluster, engine.size, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Engine Size")

p3 = ggplot(data, aes(cluster, compression.ratio, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Compression Ratio")

p4 = ggplot(data, aes(cluster, horsepower, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Horsepower")

p5 = ggplot(data, aes(cluster, peak.rpm, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Peak RPM")

p6 = ggplot(data, aes(cluster, curb.weight, color = factor(cluster))) +
  geom_boxplot() +
  labs(x = "Cluster", y = "Curb Weight")

p1 = p1 + theme(legend.position = "none")
p2 = p2 + theme(legend.position = "none")
p3 = p3 + theme(legend.position = "none")
p4 = p4 + theme(legend.position = "none")
p5 = p5 + theme(legend.position = "none")
p6 = p6 + theme(legend.position = "none")

grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3, 
             top = "Distribution of Numeric Features Across Clusters")

p7 = ggplot(data, aes(aspiration, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Aspiration", y = "Cluster") + labs(color = "Cluster") 

p8 = ggplot(data, aes(fuel.type, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Fuel Type", y = "Cluster")

p9 = ggplot(data, aes(body.style, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Body Style", y = "Cluster")

p10 = ggplot(data, aes(engine.type, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Engine Type", y = "Cluster")

p11 = ggplot(data, aes(fuel.system, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Fuel System", y = "Cluster")

p12 = ggplot(data, aes(drive.wheels, cluster, color = factor(cluster))) +
  geom_jitter() +
  labs(x = "Drive Wheels", y = "Cluster")

p7 = p7 + theme(legend.position = "none")
p8 = p8 + theme(legend.position = "none")
p9 = p9 + theme(legend.position = "none")
p10 = p10 + theme(legend.position = "none")
p11 = p11 + theme(legend.position = "none")
p12 = p12 + theme(legend.position = "none")

grid2 = grid.arrange(p8, p9, p10, p11, p12, p7, ncol = 3, 
                     top = "Distribution of Categorical Features Across Clusters")

legend = cowplot::get_legend(p7 + theme(legend.position = "bottom"))

grid.arrange(grid2, legend, ncol = 1, heights = c(8, 1))



#4. Assign mileage classes for cars in each cluster based on mileage quartiles

quartiles = by(data$avg_mpg, data$cluster, FUN = quantile, probs = c(0.25, 0.75))

# Function to classify mileage for each car
classify_mpg = function(cluster, mpg) {
  q1 = quartiles[[as.character(cluster)]]["25%"]
  q3 = quartiles[[as.character(cluster)]]["75%"]
  ifelse(mpg <= q1, "poor", ifelse(mpg <= q3, "average", "good"))
}

data$mileage_class = mapply(classify_mpg, data$cluster, data$avg_mpg)

scaled_data$cluster = data$cluster
scaled_data$mileage_class = data$mileage_class

#data = data[, !(names(data) %in% c("avg_mpg", "cluster"))]
scaled_data = scaled_data[, !(names(scaled_data) %in% c("avg_mpg", "cluster"))]


#######EDA for classification Model####

#Plot data distributions, relationship among variables etc. - refer to midterm project

#1. Univariate Analysis

#Cars made by each manufacturer
top_makes = head(sort(table(data$make), decreasing = TRUE), 10)
top_makes_df = data.frame(make = names(top_makes), count = as.numeric(top_makes))
ggplot(top_makes_df, aes(x = reorder(make, -count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(x = "Manufacturer", y = "Number of Cars", title = "Number of Cars by Manufacturer") +
  theme_minimal() +
  coord_flip() 


#Average mileage across different manufacturers
avg_mpg_by_make = aggregate(avg_mpg ~ make, data = data, FUN = mean)

avg_mpg_by_make$make <- factor(
  avg_mpg_by_make$make,
  levels = avg_mpg_by_make$make[order(avg_mpg_by_make$avg_mpg, decreasing = TRUE)]
)

ggplot(avg_mpg_by_make, aes(x = make, y = avg_mpg)) +
  geom_bar(stat = "identity", fill = "skyblue", alpha = 0.7) +
  labs(x = "Manufacturer", y = "Average mileage", title = "Average Mileage by Manufacturer") +
  theme_minimal() +
  coord_flip() 

#Average mileage across fuel types
avg_mileage = aggregate(avg_mpg ~ fuel.type, data = data, FUN = mean)
ggplot(avg_mileage, aes(x = fuel.type, y = avg_mpg, fill = fuel.type)) +
  geom_bar(stat = "identity", alpha = 0.7) +
  labs(x = "Fuel Type", y = "Average Mileage", title = "Average Mileage by Fuel Type") +
  theme_minimal()



########TRAIN CLASSIFICATION MODEL TO PREDICT MILEAGE CLASS###########

#1. Feature Selection

num_fields = c(4,8,9,10,11,12,14,15,17,18,19,20,21)
cat_fields = c(1,2,3,5,6,7,13,16,22)

for (col in num_fields) {
  scaled_data[[col]] = as.numeric(scaled_data[[col]])
}

for (col in cat_fields) {
  scaled_data[[col]] = as.factor(scaled_data[[col]])
}

#1.1 Check correlation among variables

quantvars = scaled_data[,num_fields]
corr_matrix=cor(quantvars)
round(corr_matrix, 2)
#write.csv(round(corr_matrix, 2), "D:/Fall 2023/MGSC661/Final Project/correlation_matrix.csv")

#1.2 Using PCA, check features that have very low relationship with mileage
pca = PCAmix(X.num,X.cat,ndim=2,graph=TRUE)

#1.3 Check feature importance score using random forest with all features
rf_mod1 = randomForest(scaled_data$mileage_class ~ ., data = scaled_data, importance = TRUE)
importance(rf_mod1)
#write.csv(importance(rf_mod1), "D:/Fall 2023/MGSC661/Final Project/feature_importance.csv")


#Final feature selection from the 3 steps above
cols_to_remove = c("body.style","engine.location","num.of.doors",
                   "length","width","height")

scaled_data = scaled_data[, !(names(scaled_data) %in% cols_to_remove)]


#2. Discriminant Analysis

#2.1 LDA with all variables directly
lda1 = lda(scaled_data$mileage_class ~ ., data = scaled_data)
lda1

#2.2 LDA with first two PCA components

scaled_data_without_target = scaled_data[, -which(names(scaled_data) == "mileage_class")]
scaled_data_dummies = as.data.frame(model.matrix(~ . - 1, data = scaled_data_without_target))
colnames(scaled_data_dummies) = make.names(colnames(scaled_data_dummies))
pca_result = prcomp(scaled_data_dummies, scale. = TRUE)

pca1 = pca_result$x[, 1]
pca2 = pca_result$x[, 2]

pca_lda_data = data.frame(
  "PCA component 1" = pca1,
  "PCA component 2" = pca2,
  mileage_class = scaled_data$mileage_class
)

lda2 = lda(factor(mileage_class) ~ ., data = pca_lda_data)

#Cross validation with LDA
error = rep(NA,30)
for (i in 1:30){
  sample = sample.split(pca_lda_data$mileage_class, SplitRatio = 0.5)
  train_set = subset(pca_lda_data, sample==TRUE)
  test_set = subset(pca_lda_data, sample==FALSE)
  lda_f = lda(factor(mileage_class) ~ ., data = train_set)
  predictions = predict(lda_f, newdata = test_set)
  conf_matrix = table(predictions$class, test_set$mileage_class)
  error[i] = 1 - sum(diag(conf_matrix)) / sum(conf_matrix)
}

cat("Accuracy using LDA is: ", 1-mean(error,na.rm = TRUE))

#Cross validation with QDA
error = rep(NA,30)
for (i in 1:30){
  sample = sample.split(pca_lda_data$mileage_class, SplitRatio = 0.5)
  train_set = subset(pca_lda_data, sample==TRUE)
  test_set = subset(pca_lda_data, sample==FALSE)
  qda_f = qda(factor(mileage_class) ~ ., data = train_set)
  predictions = predict(qda_f, newdata = test_set)
  conf_matrix = table(predictions$class, test_set$mileage_class)
  error[i] = 1 - sum(diag(conf_matrix)) / sum(conf_matrix)
}
cat("Accuracy using QDA is: ", 1-mean(error,na.rm = TRUE))


#Partition Matrix for LDA and QDA

partimat(factor(mileage_class) ~ ., data = pca_lda_data, method = "lda", 
                          image.colors = c("light grey", "light green", "white"),
         main = "Partition Plot for Classification using LDA")

partimat(factor(mileage_class) ~ ., data = pca_lda_data, method = "qda", 
         image.colors = c("light grey", "light green", "white"),
         main = "Partition Plot for Classification using QDA")

#3. Random Forest

scaled_data_dummies$mileage_class = scaled_data$mileage_class

rf_mod2 = randomForest(scaled_data$mileage_class ~ ., data = scaled_data, importance = TRUE, cp=0.01, na.action=na.omit)
rf_mod2

#Cross Validation for Random Forest
accuracy = rep(NA,30)
for (i in 1:30){
  sample = sample.split(scaled_data_dummies$mileage_class, SplitRatio = 0.5)
  train_set = subset(scaled_data_dummies, sample==TRUE)
  test_set = subset(scaled_data_dummies, sample==FALSE)
  rf_mod3 = randomForest(train_set$mileage_class ~ ., data = train_set, importance = TRUE, cp=0.01, na.action=na.omit)
  predicted_labels = predict(rf_mod3, newdata = test_set, type="response")
  accuracy[i] = mean(predicted_labels == test_set$mileage_class)
}

cat("Accuracy using Random Forest is: ", mean(accuracy,na.rm = TRUE))

#4. Boosting algorithm
set.seed(123)
gbm_mod = gbm(factor(mileage_class) ~ ., data = train_set, distribution = "multinomial", n.trees=10000, interaction.depth=4)

#Cross Validation for GBM
set.seed(123)
sample = sample.split(scaled_data_dummies$mileage_class, SplitRatio = 0.5)
train_set = subset(scaled_data_dummies, sample==TRUE)
test_set = subset(scaled_data_dummies, sample==FALSE)
gbm_mod2 = gbm(factor(mileage_class) ~ ., data = train_set, distribution = "multinomial", n.trees=10000, interaction.depth=4)
predicted_classes = suppressMessages(predict(gbm_mod2, newdata = test_set, type="response"))
predicted_labels = apply(predicted_classes, 1, which.max)
class_mapping = c("average", "good", "poor")
predicted_labels = class_mapping[predicted_labels]
accuracy = mean(predicted_labels == test_set$mileage_class)


#Mean accuracy using GBM
cat("Accuracy using GBM is: ", accuracy)


#####Random Forest Model Insights#####
set.seed(123)
rf_mod_final = randomForest(factor(mileage_class) ~ ., data = scaled_data, importance = TRUE, cp=0.01, na.action=na.omit)
rf_mod_final

importance(rf_mod_final)

imp = rf_mod_final$importance
ord = order(imp[, 4], decreasing = TRUE)
top10_accuracy = imp[ord[1:10], "MeanDecreaseAccuracy"]
barplot(top10_accuracy, main = "Feature Importance Scores of Attributes", xlab = "Features", ylab = "Feature Importance", col="skyblue")