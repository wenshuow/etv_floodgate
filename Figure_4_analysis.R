J = 100
CV = 10
model = 'hier'
lam = 50
use_ideal_data = FALSE
source("hiernet_source.R")
source("gender_df_president.R")
source("conjoint_prep.R")
random_seed = 67

selection = "CV" # choose from "Naive" and "CV"
if(selection == "CV"){
  # Running time is ~25 minutes (per seed)
  res = floodgate_CV(df_org, J, CV, seed = random_seed, distances = c(0:10)/1000, choose_dist = 'validation', parallel = FALSE, loss_cache = FALSE)
} else{
  # Running time is ~25 minutes (per seed)
  res = floodgate_CV(df_org, J, CV, seed = random_seed, distances = c(0)/1000, choose_dist = 'all', parallel = FALSE, loss_cache = FALSE)
}

res = as.data.frame(res)
colnames(res) = c("threshold", 'mean', 'var', 'lower confidence bound of ETV', 'p_value', 'seed', 'best_lam_sex', 'best_lam_no_sex', 'time')
res['point estimate of ETV'] = 2-2*res$mean
res[c('lower confidence bound of ETV', 'point estimate of ETV')]