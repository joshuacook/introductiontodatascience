housing_df = read.csv('data/train.csv')
test_df = read.csv('data/test.csv')

rownames(housing_df) <- housing_df$Id 
housing_df$Id <- NULL
rownames(test_df) <- test_df$Id 
test_df$Id <- NULL

housing_df$MSSubClass <- as.factor(housing_df$MSSubClass)
housing_df$OverallQual <- as.factor(housing_df$OverallQual)
housing_df$OverallCond <- as.factor(housing_df$OverallCond)
housing_df$BsmtFullBath <- as.factor(housing_df$BsmtFullBath)
housing_df$BsmtHalfBath <- as.factor(housing_df$BsmtHalfBath)
housing_df$FullBath <- as.factor(housing_df$FullBath)
housing_df$HalfBath <- as.factor(housing_df$HalfBath)
housing_df$BedroomAbvGr <- as.factor(housing_df$BedroomAbvGr)
housing_df$KitchenAbvGr <- as.factor(housing_df$KitchenAbvGr)
housing_df$TotRmsAbvGrd <- as.factor(housing_df$TotRmsAbvGrd)
housing_df$Fireplaces <- as.factor(housing_df$Fireplaces)
housing_df$GarageCars <- as.factor(housing_df$GarageCars)
housing_df$MoSold <- as.factor(housing_df$MoSold)

test_df$MSSubClass <- as.factor(test_df$MSSubClass)
test_df$OverallQual <- as.factor(test_df$OverallQual)
test_df$OverallCond <- as.factor(test_df$OverallCond)
test_df$BsmtFullBath <- as.factor(test_df$BsmtFullBath)
test_df$BsmtHalfBath <- as.factor(test_df$BsmtHalfBath)
test_df$FullBath <- as.factor(test_df$FullBath)
test_df$HalfBath <- as.factor(test_df$HalfBath)
test_df$BedroomAbvGr <- as.factor(test_df$BedroomAbvGr)
test_df$KitchenAbvGr <- as.factor(test_df$KitchenAbvGr)
test_df$TotRmsAbvGrd <- as.factor(test_df$TotRmsAbvGrd)
test_df$Fireplaces <- as.factor(test_df$Fireplaces)
test_df$GarageCars <- as.factor(test_df$GarageCars)
test_df$MoSold <- as.factor(test_df$MoSold)