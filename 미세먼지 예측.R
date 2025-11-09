# ========================
# [1] 패키지 설치 및 로드
# ========================
install.packages(c("readr", "dplyr", "lubridate", "caret", "rpart", "FNN", "ggplot2"))
library(readr)
library(dplyr)
library(lubridate)
library(caret)
library(rpart)
library(FNN)
library(ggplot2)
library(tibble)

# ========================
# [2] 데이터 불러오기 및 전처리
# ========================
# 파일 경로는 필요에 따라 수정
pollution <- read_csv("C:/Users/82108/Desktop/데마/south-korean-pollution-data.csv")
weather <- read_csv("C:/Users/82108/Desktop/데마/기상 정보(20~22).csv",
                    locale = locale(encoding = "cp949"))
geo <- read_csv("C:/Users/82108/Desktop/데마/고도와 해안 정보.csv",
                locale = locale(encoding = "cp949"))


# 월 정보 파생
pollution <- pollution %>% mutate(month = format(date, "%Y-%m"))


# 한글 District 열로 영문으로 바꾸기
weather <- weather %>%
  rename(District = 지역) %>%
  mutate(District = recode(District,
                           "서울" = "Seoul", "경기도" = "Gyeonggi", "강원도" = "Gangwon",
                           "충청북도" = "Chungbuk", "충청남도" = "Chungnam",
                           "전라북도" = "Jeonbuk", "전라남도" = "Jeonnam",
                           "경상북도" = "Gyeongbuk", "경상남도" = "Gyeongnam"
  ))

# 월 정보 생성
weather <- weather %>%
  mutate(date = as.Date(paste0(일시, "-01")),
         month = format(date, "%Y-%m")) %>%
  select(District, month,
         평균기온 = `평균기온(℃)`,
         풍속 = `평균풍속(m/s)`,
         습도 = `평균상대습도(%)`)


# 병합 수행
poll_weather <- pollution %>%
  select(-starts_with("평균기온"), -starts_with("풍속"), -starts_with("습도"),
         -starts_with("고도"), -starts_with("해안지역여부")) %>%
  left_join(weather, by = c("District", "month")) %>%
  left_join(geo, by = c("District" = "Province"))

glimpse(poll_weather)
names(poll_weather)

# 정규화
normalize <- function(x) {
  (x - min(x, na.rm = TRUE)) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
}
# ========================
# [3] 실험군 구성
# ========================
df_A <- poll_weather %>% select(pm25, 평균기온, 풍속, 습도) %>% na.omit()
df_B <- poll_weather %>% select(pm25, 평균기온, 풍속, 습도, 고도, 해안지역여부) %>% na.omit()

df_A_scaled <- df_A %>% mutate(across(.cols = -pm25, .fns = normalize))
df_B_scaled <- df_B %>% mutate(across(.cols = -pm25, .fns = normalize))

 # ========================
# [4] 성능 평가 함수 정의
# ========================
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

normalize <- function(x) (x - min(x)) / (max(x) - min(x))
# ========================
# [5] 모델 학습 및 평가 - A군
# ========================
set.seed(42)
idx_A <- createDataPartition(df_A$pm25, p = 0.8, list = FALSE)
train_A <- df_A[idx_A, ]; test_A <- df_A[-idx_A, ]

lm_A <- lm(pm25 ~ ., data = train_A)
pred_lm_A <- predict(lm_A, test_A)
rmse_lm_A <- rmse(test_A$pm25, pred_lm_A)
mae_lm_A <- mae(test_A$pm25, pred_lm_A)

tree_A <- rpart(pm25 ~ ., data = train_A)
pred_tree_A <- predict(tree_A, test_A)
rmse_tree_A <- rmse(test_A$pm25, pred_tree_A)
mae_tree_A <- mae(test_A$pm25, pred_tree_A)

idx_A <- createDataPartition(df_A_scaled$pm25, p = 0.8, list = FALSE)
train_A_scaled <- df_A_scaled[idx_A, ]; test_A_scaled <- df_A_scaled[-idx_A, ]

knn_A <- knn.reg(train = train_A_scaled[, -1], test = test_A_scaled[, -1],
                 y = train_A_scaled$pm25, k = 5)
rmse_knn_A <- rmse(test_A_scaled$pm25, knn_A$pred)
mae_knn_A <- mae(test_A_scaled$pm25, knn_A$pred)

# ========================
# [6] 모델 학습 및 평가 - B군
# ========================
set.seed(42)
idx_B <- createDataPartition(df_B$pm25, p = 0.8, list = FALSE)
train_B <- df_B[idx_B, ]; test_B <- df_B[-idx_B, ]

lm_B <- lm(pm25 ~ ., data = train_B)
pred_lm_B <- predict(lm_B, test_B)
rmse_lm_B <- rmse(test_B$pm25, pred_lm_B)
mae_lm_B <- mae(test_B$pm25, pred_lm_B)

tree_B <- rpart(pm25 ~ ., data = train_B)
pred_tree_B <- predict(tree_B, test_B)
rmse_tree_B <- rmse(test_B$pm25, pred_tree_B)
mae_tree_B <- mae(test_B$pm25, pred_tree_B)

idx_B <- createDataPartition(df_B_scaled$pm25, p = 0.8, list = FALSE)
train_B_scaled <- df_B_scaled[idx_B, ]; test_B_scaled <- df_B_scaled[-idx_B, ]

knn_B <- knn.reg(train = train_B_scaled[, -1], test = test_B_scaled[, -1],
                 y = train_B_scaled$pm25, k = 5)
rmse_knn_B <- rmse(test_B_scaled$pm25, knn_B$pred)
mae_knn_B <- mae(test_B_scaled$pm25, knn_B$pred)

# ========================
# [7] 성능 비교 테이블 출력
# ========================
result <- tibble(
  모델 = c("선형회귀", "결정트리", "KNN"),
  A_RMSE = c(rmse_lm_A, rmse_tree_A, rmse_knn_A),
  A_MAE = c(mae_lm_A, mae_tree_A, mae_knn_A),
  B_RMSE = c(rmse_lm_B, rmse_tree_B, rmse_knn_B),
  B_MAE = c(mae_lm_B, mae_tree_B, mae_knn_B)
)

print(result)

# ========================
# [8] Boxplot 시각화 - 예측 오차 비교
# ========================
boxplot_df <- bind_rows(
  tibble(모델 = "선형회귀", 군 = "A", 오차 = pred_lm_A - test_A$pm25),
  tibble(모델 = "선형회귀", 군 = "B", 오차 = pred_lm_B - test_B$pm25),
  tibble(모델 = "결정트리", 군 = "A", 오차 = pred_tree_A - test_A$pm25),
  tibble(모델 = "결정트리", 군 = "B", 오차 = pred_tree_B - test_B$pm25),
  tibble(모델 = "KNN", 군 = "A", 오차 = knn_A$pred - test_A$pm25),
  tibble(모델 = "KNN", 군 = "B", 오차 = knn_B$pred - test_B$pm25)
)

ggplot(boxplot_df, aes(x = interaction(모델, 군), y = 오차, fill = 군)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "예측 오차 Boxplot: 모델별 A군 vs B군 비교",
       x = "모델-실험군", y = "예측 오차 (예측값 - 실제값)") +
  theme_minimal()

# ========================
# [9] 결정트리 변수 중요도 시각화
# ========================
importance <- as.data.frame(varImp(tree_B)) %>%
  rownames_to_column("변수") %>%
  arrange(desc(Overall))

ggplot(importance, aes(x = reorder(변수, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "결정트리 변수 중요도 (B군 기준)",
       x = "변수", y = "중요도") +
  theme_minimal()

# ========================
# [10] 산점도 시각화 - 변수별 PM2.5 영향
# ========================
# 평균기온
ggplot(na.omit(poll_weather), aes(x = 평균기온, y = pm25)) +
  geom_point(alpha = 0.3, color = "steelblue") +
  geom_smooth(method = "lm", color = "darkred", se = TRUE) +
  labs(title = "평균기온 vs PM2.5 산점도", x = "평균기온 (°C)", y = "PM2.5 농도 (㎍/㎥)") +
  theme_minimal()

# 해안지역 여부에 따른 평균 풍속 비교
ggplot(na.omit(poll_weather), aes(x = factor(해안지역여부), y = 풍속)) +
  geom_boxplot(fill = "lightblue", alpha = 0.7) +
  labs(title = "해안지역 여부에 따른 풍속 분포 비교",
       x = "해안지역 여부 (0 = 내륙, 1 = 해안)",
       y = "평균 풍속 (m/s)") +
  theme_minimal()

# 해안지역 여부에 따른 PM2.5 농도 비교
ggplot(na.omit(poll_weather), aes(x = factor(해안지역여부), y = pm25)) +
  geom_boxplot(fill = "lightcoral", alpha = 0.7) +
  labs(title = "해안지역 여부에 따른 PM2.5 분포 비교",
       x = "해안지역 여부 (0 = 내륙, 1 = 해안)",
       y = "PM2.5 농도 (㎍/㎥)") +
  theme_minimal()

# 풍속
ggplot(na.omit(poll_weather), aes(x = 풍속, y = pm25)) +
  geom_point(alpha = 0.3, color = "forestgreen") +
  geom_smooth(method = "lm", color = "darkgreen", se = TRUE) +
  labs(title = "풍속 vs PM2.5 산점도", x = "풍속 (m/s)", y = "PM2.5 농도 (㎍/㎥)") +
  theme_minimal()

# 습도
ggplot(na.omit(poll_weather), aes(x = 습도, y = pm25)) +
  geom_point(alpha = 0.3, color = "darkorange") +
  geom_smooth(method = "lm", color = "orangered", se = TRUE) +
  labs(title = "습도 vs PM2.5 산점도", x = "상대습도 (%)", y = "PM2.5 농도 (㎍/㎥)") +
  theme_minimal()



