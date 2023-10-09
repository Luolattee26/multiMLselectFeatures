library(tidyverse)
library(ggplot2)
library(viridis)
library(ggvenn)
library(pROC)
library(RColorBrewer)
library(patchwork)
library(caret)
library(randomForest)
library(glmnet)
library(xgboost)

rm(list = ls())

# 读取数据
load(file = "MLinputdata.Rda")
data<-data[,-c(38,39)]
data%>%
  mutate(Grade=sample(LETTERS[1:6],452,replace=T))->data
data$Event<-as.factor(data$Event)

data$Grade<-as.factor(data$Grade)
str(data)
# 对两个分类变量进行独热编码
encoded_vars <- model.matrix(~Grade-1, data = data)

# 将编码后的变量添加到原始数据框中
encoded_df <- cbind(data[, !names(data) %in% c("Grade")], encoded_vars)
str(encoded_df)
# 将数据分为训练集和测试集
set.seed(123)
trainIndex <- createDataPartition(encoded_df$Event, p = .7, 
                                  list = FALSE, 
                                  times = 1)
trainData<-encoded_df[ trainIndex,]
testData<-encoded_df[-trainIndex,]

# 随机森林
fit.rf <-randomForest(Event ~ ., data =trainData, importance = TRUE, ntree = 500)
# fit.rf
# Call:
#   randomForest(formula = Event ~ ., data = trainData, importance = TRUE,      ntree = 500) 
# Type of random forest: classification
# Number of trees: 500
# No. of variables tried at each split: 6
# 
# OOB estimate of  error rate: 22.4%
# Confusion matrix:
#   0 1 class.error
# 0 246 2 0.008064516
# 1  69 0 1.000000000
#10折交叉验证
set.seed(123)
ctrl <- trainControl(method = "cv", number = 10)
rf_model_cv <- train(Event ~ ., data =trainData, method = "rf", trControl = ctrl)
#模型评价
rf_pred <- predict(rf_model_cv, newdata =testData,type="prob")[,2]
rf_roc<-roc(testData$Event,rf_pred)
auc_value <- auc(rf_roc)
p1<-ggroc(rf_roc,legacy.axes = TRUE,size=1,color="#69b3a2")+
  ggtitle("ROC curve of Randomforest")+
  geom_abline(intercept = 0,slope = 1,linetype="dashed",color="grey")+
  theme_bw()+
  annotate(geom = "text",label=paste0("AUC in test set:",round(auc_value,3)),
           x=0.7,y=0.05)
ggsave("RFroc.png",plot =p1,width = 4,height = 4,units = 'in',dpi = 600)

varImpPlot(fit.rf)
rf.imp <- importance(fit.rf)%>%as.data.frame()
rf.imp%>%
  mutate(Variable=rownames(rf.imp))%>%
  arrange(desc(MeanDecreaseAccuracy))->rf.imp

top_vars<-rf.imp[1:15,]
p2<-ggplot(top_vars, aes(y = reorder(Variable,-MeanDecreaseAccuracy), x =MeanDecreaseAccuracy)) +
  geom_bar(stat = "identity", fill = "#69b3a2",alpha=0.8,width = 0.6) +
  scale_fill_viridis() +
  labs(title = "Variable Importance Plot of RF",
       x = "Importance Score",
       y = "") +
  coord_flip()+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 30,hjust =0.9))
ggsave("RFvip.png",plot =p2,width =6,height = 4,units = 'in',dpi = 600)
plot(fit.rf)
# 提取树与OOB的关系的数值
err_rates <-fit.rf[["err.rate"]]
err_rates%>%
  as.data.frame()%>%
  mutate(Tree=1:500)->err_rates
err_rates%>%
  pivot_longer(cols = 1:3,
               names_to = "OOB",
               values_to = "value")->err_rates

p3<-ggplot(err_rates, aes(x=Tree, y=value,color=OOB)) +
  geom_line(size=1) +
  labs(title = "The relationship between tree number and OOB",
       x="Number of Trees", y="Error Rate") +
  scale_color_brewer(palette = "Set2",
                     name="Error rate",
                     label=c("Alive","Death","OOB"))+
  theme_bw()
ggsave("RFoob.png",plot =p3,width =6,height = 4,units = 'in',dpi = 600)
p1+(p2/p3)&
  theme(plot.title = element_text(size =12))
ggsave("RF.png",plot =last_plot(),width =9,height =4.5,units = 'in',dpi = 600)  

# Lasso回归
# LASSO回归模型
set.seed(1234)
x <- model.matrix(Event~ ., trainData)[,-1]
y <- as.numeric(trainData$Event)-1
fit.lasso<-glmnet(x, y, alpha = 1,
                  family = "binomial",
                  type.measure = "class")
cv_lasso<- cv.glmnet(x, y, alpha =1,
                     family = "binomial",
                     type.measure = "class",
                     nfolds = 10)
#变量筛选过程
plot(fit.lasso,xvar="lambda")
# 提取plot(fit.lasso)中的数据
plot.data <- data.frame(as.matrix(fit.lasso$lambda), as.matrix(t(fit.lasso$beta)))
plot.data%>%
  rename(lambda='as.matrix.fit.lasso.lambda.')%>%
  pivot_longer(cols = 2:ncol(plot.data),
               names_to ="variable",
               values_to = "coef")->plot.data

# 绘制图形
# 使用viridis调色板
colors <- viridis(42)
scientific_10 <- function(x) {
  parse(text=gsub("e"," %*% 10^", scales::scientific_format()(x)))
}
p4<-ggplot(plot.data, aes(x = lambda, y = coef, color = variable)) +
  geom_line(size=1,alpha=0.8) +
  scale_color_manual(values =colors)+
  scale_x_log10(label=scientific_10)+
  labs(title = "LASSO Regression Path",
       x = "Log lambda",
       y = "Coefficient") +
  theme_bw()+
  theme(legend.text = element_text(size =6))
ggsave("lasso1.png",plot =p4,width =8,height =6,units = 'in',dpi = 600)  

plot(cv_lasso)

# ROC曲线模型评价
x_test <- model.matrix(Event ~ ., testData)[,-1]
y_test <- as.numeric(testData$Event) - 1
y_pred <- predict(cv_lasso, s = cv_lasso$lambda.min, newx = x_test, type = "class")
y_prob <- predict(cv_lasso, s = cv_lasso$lambda.min, newx = x_test, type = "response")
roc_lasso<- roc(y_test, y_prob[,1])
auc_value <- auc(roc_lasso)
p5<-ggroc(roc_lasso,legacy.axes = TRUE,size=1,color="#69b3a2")+
  ggtitle("ROC curve of LASSO")+
  geom_abline(intercept = 0,slope = 1,linetype="dashed",color="grey")+
  theme_bw()+
  annotate(geom = "text",label=paste0("AUC in test set:",round(auc_value,3)),
           x=0.7,y=0.05)
ggsave("lassoroc.png",plot =p5,width = 4,height = 4,units = 'in',dpi = 600)

cv_lasso
# Call:  cv.glmnet(x = x, y = y, type.measure = "class", nfolds = 10,      alpha = 1, family = "binomial") 
# 
# Measure: Misclassification Error 
# 
# Lambda Index Measure      SE Nonzero
# min 0.01384    17  0.2145 0.02590      19
# 1se 0.06133     1  0.2177 0.02447       0
##min模型--提取简洁模型的参数系数
se_lambda<-cv_lasso$lambda.min #lambda.min对应的lambda值
se_coef<-coef(cv_lasso, s = "lambda.min")##λ=最小值时各变量的回归系数值
se_coef

index<-which(se_coef!=0)#非零系数
coef<-se_coef[index][-1]#对应回归系数
diffvariables=row.names(se_coef)[index][-1]#非零变量
lasso.result.se<-cbind(diffvariables,coef)%>%
  as.data.frame()#输出结果
lasso.result.se%>%
  mutate(direction=ifelse(coef>0,"Up","Down"))->lasso.result.se
lasso.result.se$coef<-as.numeric(lasso.result.se$coef)

p6<-ggplot(lasso.result.se, aes(y=reorder(diffvariables,coef),x=coef,fill=direction)) +
  geom_col(alpha=0.8,width = 0.6) +
  labs(title = "Important variables in LASSO",
       x = "Coef",
       y = "") +
  coord_flip()+
  scale_fill_brewer(palette = "Set2")+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 30,hjust =0.9),
        legend.position = "")
ggsave("lassovip.png",plot =p6,width =6,height = 4,units = 'in',dpi = 600)

# XGBoost
# 将数据转换为DMatrix格式
trainlabel <-as.numeric(trainData$Event)-1
testlabel <-as.numeric(testData$Event)-1
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -1]), 
                            label =trainlabel)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -1]), 
                           label =testlabel)

# 定义参数
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.01,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# 10折交叉验证
set.seed(123)
cv <- xgb.cv(params = params, 
             data = train_matrix, 
             nrounds = 1000, 
             nfold = 10, 
             early_stopping_rounds = 10)

# 找到最佳迭代次数
best_iter <- which.max(cv$evaluation_log$test_auc_mean)

# 训练模型
model <- xgb.train(params = params, 
                   data = train_matrix, 
                   nrounds = best_iter)

# 预测测试集
pred <- predict(model, test_matrix)

# 计算ROC曲线和AUC值
roc_xgb<- roc(testData$Event, pred)
auc <- auc(roc_xgb)
p7<-ggroc(roc_xgb,legacy.axes = TRUE,size=1,color="#69b3a2")+
  ggtitle("ROC curve of XGBoost")+
  geom_abline(intercept = 0,slope = 1,linetype="dashed",color="grey")+
  theme_bw()+
  annotate(geom = "text",label=paste0("AUC in test set:",round(auc,3)),
           x=0.7,y=0.05)
ggsave("xgboostroc.png",plot =p7,width = 4,height = 4,units = 'in',dpi = 600)
# 变量重要性评价
importance <- xgb.importance(model = model)
importance <- importance[order(importance$Gain, decreasing = TRUE), ][1:15,1:2]

p8<-ggplot(importance, aes(y = reorder(Feature,-Gain), x =Gain)) +
  geom_bar(stat = "identity", fill = "#69b3a2",alpha=0.8,width = 0.6) +
  scale_fill_viridis() +
  labs(title = "Variable Importance Plot of XGBoost",
       x = "Importance Score",
       y = "") +
  coord_flip()+
  theme_bw()+
  theme(axis.text.x = element_text(angle = 30,hjust =0.9))
ggsave("xgboostvip.png",plot =p8,width =6,height = 4,units = 'in',dpi = 600)

source("shap.R")
shap_result = shap.score.rank(xgb_model = model, 
                              X_train =train_matrix,
                              shap_approx = F)

var_importance(shap_result, top_n=10)
#计算前10个特征的SHAP值
shap_long_hd = shap.prep(X_train =trainData, top_n = 10)

#SHAP值可视化
plot.shap.summary(data_long = shap_long_hd)
ggsave("xgboostshap.png",plot =last_plot(),width =6,height =5,units = 'in',dpi = 600)


hubgene<-list(
  RF=rownames(top_vars),
  LASSO=lasso.result.se$diffvariables,
  XGBoost=importance$Feature
)

ggvenn(hubgene,
       show_percentage = F,#不显示百分比
       stroke_color ="black",#线圈边框色
       stroke_alpha =0.3,#透明度
       stroke_linetype ="solid",#线型
       stroke_size =0.8,#线粗细
       set_name_color ="red",#每个元素文本颜色
       set_name_size =6,#文本大小
       text_color = "white",#韦恩图中数字颜色
       text_size =6)+#数字大小
  scale_fill_brewer(palette = "Set1")#填充色
ggsave("vnn.png",plot =last_plot(),width =6,height =6,units = 'in',dpi = 600)

hubgenename<-intersect(intersect(rownames(top_vars),lasso.result.se$diffvariables),
                       importance$Feature)
hubgenename
# [1] "ABCA7" "ABCA5"