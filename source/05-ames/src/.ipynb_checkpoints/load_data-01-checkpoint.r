housing_df = read.csv('../data/housing.csv')

rownames(housing_df) <- housing_df$Id 
housing_df$Id <- NULL

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