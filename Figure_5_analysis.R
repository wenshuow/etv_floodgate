# Running time is ~20 minutes (per seed)
J = 100
CV = 10
model = 'hier'
lam = 50
pr = 0.9 # choose pr in (0.5, 1]
use_ideal_data = TRUE
source("hiernet_source.R")
source("gender_df_president.R")
source("conjoint_prep.R")

res = floodgate_CV(df_org, J, CV, seed = 67, distances = c(0)/1000, choose_dist = 'all', parallel = FALSE, loss_cache = FALSE)
res = as.data.frame(res)
colnames(res) = c("threshold", 'mean', 'var', 'floodgate lower bound', 'p_value', 'seed', 'best_lam_sex', 'best_lam_no_sex', 'time')
res['theoretical upper bound'] = (1-0.2726008) * (pr-0.5)
res[c('floodgate lower bound', 'theoretical upper bound')]