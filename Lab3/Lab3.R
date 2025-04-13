data <- read.csv("./income.data.csv")

model1 <- lm(happiness ~ income, data = data)

summary(model1)


plot(data$income, data$happiness,
     main = "Income vs Happiness",
     xlab = "Income (USD)", ylab = "Happiness",
     pch = 19, col = "blue")
abline(model1, col = "red", lwd = 2)

# Happiness = 0.20427 + 0.71383 * Income
# R2 = 0.7493 meaning that percentage is the ammount of variation that can be explained by income
# with a 0.71383 slope every 1 unit more that is increased in income the the happiness should go up by this amount

residuals1 <- residuals(model1)

plot(residuals1, main = "Residuals of Simple Linear Regression",
     ylab = "Residuals")

RSS1 <- sum(residuals1^2)
cat("Residual Sum of Squares (RSS):", RSS1, "\n")

MSE1 <- mean(residuals1^2)
cat("Mean Squared Error (MSE):", MSE1, "\n")

#The risiduals tell me that it would be a good fit as there is no clear pattern emerging
# 255.7715, indicates that there is a lot of variance in data or the model isnt capturing the trend well
# 0.5135974, indicated that the model is off by that amount in its predictive accuracy