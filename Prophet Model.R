## PROPHET MODEL FOR DENGUE FORECASTING

## Load Libraries
library(prophet)
library(readr)
library(tidyverse)
library(Metrics)

## LOAD DATA

dengue_data <- read_csv(
  "C:/Users/ADMIN/Desktop/Time Series Session/Datasets/Dengue.csv"
)

## DATA EXPLORATION

## View structure
glimpse(dengue_data)

## Summary statistics
summary(dengue_data)

## Missing values
colSums(is.na(dengue_data))

## DATA PREPARATION

# Convert date column
dengue_data$ds <- as.Date(dengue_data$ds)

## Ensure target variable is numeric
dengue_data$y <- as.numeric(dengue_data$y)

## Sort by date
dengue_data <- dengue_data %>%
  arrange(ds)

# VISUALIZE HISTORICAL DATA

ggplot(dengue_data,
       aes(x = ds,
           y = y)) +
  geom_line(color = "steelblue") +
  labs(
    title = "Historical Daily Dengue Cases",
    x = "Date",
    y = "Cases"
  ) +
  theme_minimal()

## Why I chose Prophet
## 1. Clear Seasonality
## Dengue cases often rise and fall during certain periods of the year.
## Long Historical Data. We have 2557 daily observations ≈ 7 years of data
## Prophet generally performs well when several years of history are available.
## 3. Trend Changes
## Disease cases can increase or decrease over time.
## Prophet automatically detects trend changes (changepoints) without requiring extensive manual modeling.
## 4. Easy Interpretation
## Prophet separates the forecast into:
## Trend
## Yearly seasonality
## Weekly seasonality
## making it easy to explain in reports and presentations.

## CHECK FOR MISSING DATES

all_dates <- seq(
  from = min(dengue_data$ds),
  to = max(dengue_data$ds),
  by = "day"
)

cat("Expected dates:", length(all_dates), "\n")
cat("Actual rows:", nrow(dengue_data), "\n")

## TRAIN-TEST SPLIT (80% TRAIN, 20% TEST)

split_index <- floor(0.80 * nrow(dengue_data))

train_data <- dengue_data[1:split_index, ]
test_data <- dengue_data[(split_index + 1):nrow(dengue_data), ]

cat("Training observations:", nrow(train_data), "\n")
cat("Testing observations:", nrow(test_data), "\n")

## The Prophet model was trained using 2,045 observations (80%) of the dataset.
## The remaining 512 observations (20%) were reserved for testing.
## This allows the model's forecasting ability to be evaluated on unseen data.

## FIT PROPHET MODEL

model <- prophet(
  yearly.seasonality = TRUE,
  weekly.seasonality = TRUE,
  daily.seasonality = FALSE,
  changepoint.prior.scale = 0.05
)

model <- fit.prophet(model, train_data)

## yearly.seasonality = TRUE → Captures yearly patterns in dengue cases.
## weekly.seasonality = TRUE → Captures weekly variations in the data.
## daily.seasonality = FALSE → Daily patterns are not modeled because the data does not require them.
## changepoint.prior.scale = 0.05 → Controls how flexible the trend is; a small value produces a smoother trend.
## fit.prophet(model, train_data) → Trains the Prophet model using the training dataset.

## FORECAST TEST PERIOD

future_test <- make_future_dataframe(
  model,
  periods = nrow(test_data),
  freq = "day"
)

forecast_test <- predict(model, future_test)

## make_future_dataframe() creates future dates for forecasting.
## periods = nrow(test_data) tells Prophet to forecast the same number of days as the test dataset (512 days).
## freq = "day" specifies that forecasts are generated daily.
## future_test contains both historical and future dates.
## predict(model, future_test) uses the trained Prophet model to generate forecasts for those dates and stores the results in forecast_test.

## EXTRACT TEST PREDICTIONS

predictions <- forecast_test %>%
  tail(nrow(test_data)) %>%
  select(ds, yhat, yhat_lower, yhat_upper)

## MODEL EVALUATION

RMSE <- rmse(
  actual = test_data$y,
  predicted = predictions$yhat
)

MAE <- mae(
  actual = test_data$y,
  predicted = predictions$yhat
)

MAPE <- mean(
  abs((test_data$y - predictions$yhat) /
        test_data$y),
  na.rm = TRUE
) * 100

R2 <- cor(test_data$y,
          predictions$yhat)^2
cat("\n")
cat("MODEL PERFORMANCE\n")
cat("-----------------\n")
cat("RMSE :", round(RMSE, 2), "\n")

## On average, the model's predictions differ from the actual dengue cases by about 3.17 cases.
## A lower RMSE indicates better predictive accuracy.
## Since dengue cases range from approximately 5 to 94 cases, an RMSE of 3.17 is very low.

cat("MAE  :", round(MAE, 2), "\n")
## The model's forecasts differ from the actual values by about 2.51 cases on average.
## This indicates that prediction errors are small and the model is highly accurate.

cat("MAPE :", round(MAPE, 2), "%\n")
## The forecasts deviate from the actual dengue cases by only 3.83% on average.
## A MAPE below 10% is generally considered highly accurate.
## Therefore, the Prophet model demonstrates excellent forecasting performance.

cat("R-squared:", round(R2,4))

## The model explains approximately 95.33% of the variation in dengue cases.
## This indicates a very strong relationship between the predicted and actual values.
## Values closer to 1 indicate better model fit.

## ACTUAL VS PREDICTED

comparison <- data.frame(
  Date = test_data$ds,
  Actual = test_data$y,
  Predicted = predictions$yhat
)

ggplot(comparison,
       aes(Date)) +
  
  geom_line(
    aes(y = Actual,
        color = "Actual")
  ) +
  
  geom_line(
    aes(y = Predicted,
        color = "Predicted")
  ) +
  
  labs(
    title = "Actual vs Predicted Dengue Cases",
    x = "Date",
    y = "Cases",
    color = ""
  ) +
  theme_minimal()

# RESIDUAL ANALYSIS

comparison$Residuals <-
  comparison$Actual -
  comparison$Predicted

ggplot(comparison,
       aes(Date, Residuals)) +
  geom_line(color = "red") +
  geom_hline(
    yintercept = 0,
    linetype = "dashed"
  ) +
  labs(
    title = "Residual Analysis",
    x = "Date",
    y = "Residuals"
  ) +
  theme_minimal()

## REFIT MODEL ON FULL DATASET

final_model <- prophet(
  dengue_data,
  yearly.seasonality = TRUE,
  weekly.seasonality = TRUE,
  daily.seasonality = FALSE,
  changepoint.prior.scale = 0.05
)

## FORECAST NEXT 365 DAYS

future <- make_future_dataframe(
  final_model,
  periods = 365,
  freq = "day"
)

forecast <- predict(
  final_model,
  future
)

## FORECAST PLOT

plot(final_model, forecast)

## FORECASTED VALUES PLOT

ggplot(forecast,
       aes(ds, yhat)) +
  geom_line(color = "darkgreen") +
  labs(
    title = "365-Day Dengue Forecast",
    x = "Date",
    y = "Predicted Cases"
  ) +
  theme_minimal()

## SEASONALITY COMPONENTS

prophet_plot_components(
  final_model,
  forecast
)

## FORECAST TABLE

forecast %>%
  select(
    ds,
    yhat,
    yhat_lower,
    yhat_upper
  ) %>%
  tail(20)

## ds - Forecast date                        
## yhat - Predicted number of dengue cases   
## yhat_lower - Lower bound of the forecast interval 
## yhat_upperUpper - bound of the forecast interval 

## Example Interpretation
## For 31 December 2026:
## The Prophet model predicts approximately 75 dengue cases on 31 December 2026. Considering forecast uncertainty, the actual number of cases is expected to lie between 71 and 79 cases.

## Overall Trend
## From the forecast:
## 12 Dec 2026 → 64.65 cases
## 31 Dec 2026 → 75.35 cases 

## This indicates that:
## Dengue cases are expected to increase gradually during the second half of December 2026.
## The forecast rises from about 65 cases to 75 cases.
## This suggests an upward trend in dengue incidence toward the end of the forecast horizon.

## Forecast Uncertainty
## Notice that: yhat_lower < yhat < yhat_upper
## The interval is relatively narrow, indicating that the model is fairly confident about its predictions.

## Report Interpretation
## The Prophet model forecasted a gradual increase in dengue cases during December 2026. Predicted daily cases ranged from approximately 65 to 75 cases, with forecast intervals indicating moderate uncertainty. By 31 December 2026, the model predicted about 75 cases, with the actual number expected to fall between 71 and 79 cases. The results suggest a continuing upward trend in dengue incidence toward the end of the forecast period.

  