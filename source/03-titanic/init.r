verify_length <- function (v1, v2 ){
    if (length(v1) != length(v2)) {
        stop('length of vectors do not match') 
    }
}

accuracy <- function (actual, predicted) {
    verify_length(actual, predicted)
    return(sum(actual == predicted)/length(actual))
}

titanic <- read.csv('titanic.csv')
number_of_passengers = length(titanic$Survived)

no_survivors <- rep(0, number_of_passengers)

women_mask = titanic$Sex == 'female'
women_survived = rep(no_survivors)
women_survived[women_mask] = 1

random_mask = sample(c(TRUE,FALSE), number_of_passengers, replace = TRUE)
random_model = rep(no_survivors)
random_model[random_mask] = 1

first_class_mask = titanic$Pclass == 1
women_and_first_class_survived = rep(women_survived)
women_and_first_class_survived[first_class_mask] = 1

children_mask = titanic$Age < 7
women_and_children_survived = rep(women_survived)
women_and_children_survived[children_mask] = 1


library(repr)
options(repr.plot.width=10, repr.plot.height=4)

