#HW1
#Lance Swett
#2/14/25

# 1.a read Su_raw_matrix.txt to variable called su turning it into a data frame
su <- read.delim("./Su_raw_matrix.txt", sep="\t", header = TRUE);

# Print the first few rows to verify the data
#head(su)

# 1.b calculates the mean and standard deviation for the liver_2.cel 
#Using the na.rm = trie to ignore any cells that are missing data
mean_liver2 <- mean(su$Liver_2.CEL, na.rm = TRUE)
sd_liver2 <- sd(su$Liver_2.CEL, na.rm = TRUE)

#print results for verification
print(paste("Mean of Liver_2.CEL:", mean_liver2))
print(paste("Standard Deviation of Liver_2.CEL:", sd_liver2))

# 1.c calculate colMeans and colSums for each column
column_means <- colMeans(su, na.rm = TRUE)
column_sums <- colSums(su, na.rm = TRUE)


#Print results of first 5 columns of each
print(paste("Column Means first five columns: ", column_means[1:5]))
print(paste("Column Sums first five columns: ", column_sums[1:5]))


# 2.a generate random numbers and create a histogram to show plots of different standard deviations
n <- 10000

dataset1 <- rnorm(n, mean=0, sd=0.2)

#opens an image file to save data to
png("hist_mean0_Signma0.2.png")

#plots the histogram
hist(dataset1, main = "Histogram (Mean = 0, Signma = 0.2)",
     xlab = "Values", col = "blue",
     xlim = c(-5,5), breaks = 50)

#save and close file
dev.off()

dataset2 <- rnorm(n, mean=0, sd=0.5)

#opens an image file to save data to
png("hist_mean0_Signma0.5.png")

#plots the histogram
hist(dataset1, main = "Histogram (Mean = 0, Signma = 0.5)",
     xlab = "Values", col = "red",
     xlim = c(-5,5), breaks = 50)

#save and close file
dev.off()

# 3 load ggplot2 library
library(ggplot2)

# 3.a create the sample data frame
set.seed(42)
dat <- data.frame(
  cond = factor(rep(c("A", "B"), each = 200)), #catagory labels
  rating = c(rnorm(200), rnorm(200, mean = 0.8)) #Normal Distrobution
)

# 3.b Overlaid histograms
ggplot(dat, aes(x = rating, fill = cond)) +
  geom_histogram(binwidth = 0.5, alpha = 0.5, position = "identity")

# 3.c Interleaved histograms
ggplot(dat, aes(x = rating, fill = cond)) +
  geom_histogram(binwidth = 0.5, position = "dodge")

# 3.d Density plots
ggplot(dat, aes(x = rating, colour = cond)) +
  geom_density()

# 3.e Density plots with semitransparent fill
ggplot(dat, aes(x = rating, fill = cond)) +
  geom_density(alpha = 0.3)

ggsave("overlaid_histogram.png")  # Save plot as image

# 3.f diabetes
diabetes <- read.csv("diabetes_train.csv")

ggplot(diabetes, aes(x = mass, fill = class)) +
  geom_histogram(binwidth = 5, alpha = 0.5, position = "identity")

ggplot(diabetes, aes(x = mass, fill = class)) +
  geom_histogram(binwidth = 5, position = "dodge")

ggplot(diabetes, aes(x = mass, fill = class)) +
  geom_density()

ggplot(diabetes, aes(x = mass, fill = class)) +
  geom_density(alpha = 0.3)

ggsave("overlaid_diabete_histogram.png")  # Save plot as image

# 4 read data into variable passengers
library(tidyverse)
passengers <- read.csv("titanic.csv")

# 4.a removes empty data cells
passengers %>% drop_na() %>% summary()

# 4.b  filters for passengers who are male
passengers %>% filter(Sex == "male")

# 4.c sorts the passengers by their fare
passengers %>% arrange(desc(Fare))

# 4.d Creates a new colum named famsize and this sums parch and sibsp
passengers %>% mutate(FamSize = Parch + SibSp)

# 4.e groups by gender and calculates the mean fare and survival count
passengers %>% group_by(Sex) %>% summarise(meanFare = mean(Fare), numSurv = sum(Survived))

# 5
diabetes2 <- read.csv("diabetes_train.csv")

skin_percentiles <- quantile(diabetes2$skin, probs = c(0.1, 0.3, 0.5, 0.6), na.rm = TRUE)
print(skin_percentiles)