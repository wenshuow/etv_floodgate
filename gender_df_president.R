# Replication Code for "Using Machine Learning to Test Causal Hypotheses in Conjoint Analysis" by Ham, Imai, and Janson. (2022)
# helper source code for reading in Gender Conjoint Analysis data from Ono and Burden (2019)

library(data.table)

load("data/POBE_R_data.RData")

col_names = names(x)

n = nrow(x)
x[col_names[4:18]] = lapply(x[col_names[4:18]] , factor)

#aligning left and right profiles
x_1 = x[x$profile == 1, ]
x_2 = x[x$profile == 2, ]
colnames(x_2) <- paste0(colnames(x_2), "_2")

gender = cbind(x_1, x_2)
#Extracting Congressional candidates
gender_pres = gender[gender$Office == "President", ]

#taking relevant factors and respondent characteristics
variable = colnames(gender)[c(4:10, 12:17, 36:42, 44:49, 22:24, 26:32)]

Y = gender_pres$selected

df = gender_pres
int_df = data.frame(df[, variable])
int_df$Y = df$selected

int_df = int_df[int_df$R_Hillary != "NA", ]
int_df = int_df[int_df$R_Partisanship != "NA", ]

int_df = na.omit(int_df)





