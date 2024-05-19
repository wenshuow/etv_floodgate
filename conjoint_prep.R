library(glmnet)
tp = "President_Party"
lambda = c(1, 10, 20, 30, 40, 50)

match_party = function(cand, resp){
  stopifnot(cand %in% c('Democratic Party', 'Republican Party'))
  stopifnot(resp %in% c('Democrat', 'Republican'))
  if(cand=='Republican Party'){
    if(resp=='Republican'){
      return(TRUE)
    } else{
      return(FALSE)
    }
  } else{
    if(resp=='Republican'){
      return(FALSE)
    } else{
      return(TRUE)
    }
  }
}
df_org = int_df
if(use_ideal_data){
  stopifnot(pr>0.5)
  for(i in 1:nrow(df_org)){
    party_1 = df_org$Party.affiliation[i]
    party_2 = df_org$Party.affiliation_2[i]
    party_r = df_org$R_Partisanship[i]
    
    if(party_1==party_2){
      df_org$Y[i] = rbinom(1, 1, 0.5)
    } else if(party_r=='Independent'){
      df_org$Y[i] = rbinom(1, 1, 0.5)
    } else{
      if(match_party(party_1, party_r) & !match_party(party_2, party_r)){
        df_org$Y[i] = rbinom(1, 1, 1-pr)
      } else if(match_party(party_2, party_r) & !match_party(party_1, party_r)){
        df_org$Y[i] = rbinom(1, 1, pr)
      } else{
        stop()
      }
    }
  }
}

fit_model <- function(X, y_var){
  if(model == 'lasso'){
    cross_val <- cv.glmnet(as.matrix(X), as.matrix(y_var), 
                           family = 'binomial', 
                           type.measure = 'deviance',
                           alpha = 1, 
                           nlambda = 100)
    fit_1se <- glmnet(as.matrix(X), as.matrix(y_var), 
                      family = 'binomial', 
                      alpha = 1, 
                      lambda = cross_val$lambda.1se)
    return(list(fit_1se, cross_val$lambda.1se))
  }
  if(model == 'hier'){
    # best_lam = get_lam(c(1, 10, 20, 30, 40, 50), X = X, y_var = y_var, seed = 9)
    best_lam = lam
    invisible(capture.output(fit <- hierNet.logistic(as.matrix(X), y_var, lam= best_lam, diagonal = FALSE,trace = 0)))
    return(list(fit, best_lam))
  }
}

get_loss <- function(fit_sex, fit_no_sex, df, distances, seed, J){
  df_raw = df
  n_i = length(distances)
  ndata = dim(df_raw)[1]
  losses = matrix(0, ndata, n_i)
  gender_data = get_data(df_raw)
  X = gender_data[[1]]
  X_no_sex = gender_data[[2]]
  y_var = gender_data[[3]]
  tilde = get_probs(fit_no_sex, X_no_sex[1:ndata,])
  real = get_probs(fit_sex, X[1:ndata,])
  pred = abs(1-y_var[1:ndata]-real)/(abs(1-y_var[1:ndata]-real)+abs(1-y_var[1:ndata]-tilde))
  for(i in 1:n_i){
    pred_i = sapply(pred, transfer_pred, distances[i], simplify = TRUE)
    losses[,i] = losses[,i] + (1-pred_i)
  }
  for(j in 1:J){
    set.seed(seed*j*2)
    if(tp=="President" || tp=="Congress"){
      df_raw['Sex'] = sample(c('Male', 'Female'), ndata, TRUE)
      df_raw['Sex'] = lapply(df_raw['Sex'], factor)
      df_raw['Sex_2'] = sample(c('Male', 'Female'), ndata, TRUE)
      df_raw['Sex_2'] = lapply(df_raw['Sex_2'], factor)
    } else{
      df_raw['Party.affiliation'] = sample(c("Republican Party", 'Democratic Party'), ndata, TRUE)
      df_raw['Party.affiliation'] = lapply(df_raw['Sex'], factor)
      df_raw['Party.affiliation_2'] = sample(c("Republican Party", 'Democratic Party'), ndata, TRUE)
      df_raw['Party.affiliation_2'] = lapply(df_raw['Sex_2'], factor)
    }
    gender_data = get_data(df_raw)
    X = gender_data[[1]]
    X_no_sex = gender_data[[2]]
    tilde = get_probs(fit_no_sex, X_no_sex[1:ndata,])
    real = get_probs(fit_sex, X[1:ndata,])
    pred = abs(1-y_var[1:ndata]-real)/(abs(1-y_var[1:ndata]-real)+abs(1-y_var[1:ndata]-tilde))
    for(i in 1:n_i){
      pred_i = sapply(pred, transfer_pred, distances[i], simplify = TRUE)
      losses[,i] = losses[,i] + pred_i/J
    }
  }
  return(losses)
}
df = int_df
first = c(1:13)
second = c(14:26)
n = nrow(df)
empty_df = df
y_new = factor(1- (as.numeric(int_df$Y) - 1))

for (i in 1:length(first)) {
  empty_df[, first[i]] = df[, second[i]]
  empty_df[, second[i]] = df[, first[i]]
}

empty_df$Y = y_new

final_df = rbind(int_df, empty_df)

col_names = colnames(final_df)

final_df[col_names[c(27, 28, 30:36)]] = lapply(final_df[col_names[c(27, 28, 30:36)]] , factor)
no_sex = final_df[c(1:6, 8:13, 14:19, 21:37)]

# Forcing interaction
X = model.matrix(Y~ . + Sex*Party.affiliation + Sex*Party.affiliation_2 +Sex_2*Party.affiliation + Sex_2*Party.affiliation_2, final_df, contrasts.arg = lapply(final_df[, c(1:28, 30:36)], contrasts, contrasts = FALSE))[, -1]
X_no_sex = model.matrix(Y~., no_sex, contrasts.arg = lapply(no_sex[, c(1:26, 28:34)], contrasts, contrasts = FALSE))[, -1]
xnames = colnames(X)
xnsnames = colnames(X_no_sex)
y_var = as.numeric(final_df$Y)- 1
transfer_pred = function(p, distance){
  if(p>0.5+distance)
    return(1)
  if(p<0.5-distance)
    return(0)
  return(0.5)
}

get_data <- function(int_df){
  # print(dim(int_df))
  # Enforcing constraints
  df = int_df
  first = c(1:13)
  second = c(14:26)
  n = nrow(df)
  empty_df = df
  y_new = factor(1- (as.numeric(int_df$Y) - 1))
  
  for (i in 1:length(first)) {
    empty_df[, first[i]] = df[, second[i]]
    empty_df[, second[i]] = df[, first[i]]
  }
  
  empty_df$Y = y_new
  
  final_df = rbind(int_df, empty_df)
  
  col_names = colnames(final_df)
  
  final_df[col_names[c(27, 28, 30:36)]] = lapply(final_df[col_names[c(27, 28, 30:36)]] , factor)
  no_sex = final_df[c(1:6, 8:13, 14:19, 21:37)]
  # Forcing interaction
  X = model.matrix(Y~ . + Sex*Party.affiliation + Sex*Party.affiliation_2 +Sex_2*Party.affiliation + Sex_2*Party.affiliation_2, final_df, contrasts.arg = lapply(final_df[, c(1:28, 30:36)], contrasts, contrasts = FALSE))[, -1]
  X_no_sex = model.matrix(Y~., no_sex, contrasts.arg = lapply(no_sex[, c(1:26, 28:34)], contrasts, contrasts = FALSE))[, -1]
  # print(dim(X))
  # print(dim(X_no_sex))
  if(length(colnames(X)) < length(xnames)){
    # print('a')
    missing = matrix(0, nrow = dim(X)[1], ncol = length(xnames) - length(colnames(X)))
    colnames(missing) = setdiff(xnames, colnames(X))
    X = cbind(X, missing)[,xnames]
  }
  if(length(colnames(X_no_sex)) < length(xnsnames)){
    # print('b')
    missing = matrix(0, nrow = dim(X_no_sex)[1], ncol = length(xnsnames) - length(colnames(X_no_sex)))
    colnames(missing) = setdiff(xnsnames, colnames(X_no_sex))
    X_no_sex = cbind(X_no_sex, missing)[,xnsnames]
  }
  
  y_var = as.numeric(final_df$Y)- 1
  return(list(X, X_no_sex, y_var))
}

get_probs <- function(fit, X){
  if(model == 'hier'){
    odds = fit$b0 + X %*% (fit$bp-fit$bn) + 0.5 * apply(X, 1, function(x) return(x%*%fit$th%*%x))
    probs = exp(odds)
    probs = probs/(1+probs)
  } else{
    probs = c(predict(fit, newx = as.matrix(X), type = 'response'))
  }
  return(probs)
}

floodgate <- function(df_train, df_validate, J, seed, icv, CV, distances, choose_dist = "validation", parallel = FALSE, loss_cache = FALSE){
  stopifnot(choose_dist == "all" || choose_dist == "validation")
  n_validate = dim(df_validate)[1]
  n_train = dim(df_train)[1]
  n_i = length(distances)
  best_lam_sex = -1
  best_lam_no_sex = -1
  gender_data = get_data(df_train)
  X = gender_data[[1]]
  X_no_sex = gender_data[[2]]
  y_var = gender_data[[3]]
  fit_no_sex = fit_model(X_no_sex, y_var)
  best_lam_no_sex = fit_no_sex[[2]]
  fit_no_sex = fit_no_sex[[1]]
  fit_sex = fit_model(X, y_var)
  best_lam_sex = fit_sex[[2]]
  fit_sex = fit_sex[[1]]
  if(choose_dist == "all"){
    chosen_distance = -1
    losses = get_loss(fit_sex, fit_no_sex, df_validate, distances, seed, J)
  } else{
    n_per_cv = floor(n_train/CV)
    cutoffs = c(1, n_per_cv * c(1:(CV-1)), n_train)
    shuffles = sample(1:n_train)
    shuffled = df_train[shuffles, ]
    if(loss_cache){
      if(parallel){
        file_dir = sprintf("models/cv/parallel_seed/%s/%s_%d/CV%d/J%d/%d",tp,model,lam,CV,J,icv)
      } else{
        file_dir = sprintf("models/cv/normal_seed/%s/%s_%d/CV%d/J%d/%d",tp,model,lam,CV,J,icv)
      }
      dir.create(file.path(file_dir), recursive = TRUE, showWarnings = FALSE)
      loss_file = sprintf("%s/loss_%d.csv",file_dir, seed)
      vars_file = sprintf("%s/var_%d.csv",file_dir, seed)
    }

    if(loss_cache & file.exists(loss_file)){
      print('exists')
      load(loss_file)
      load(vars_file)
    } else{
      losses = rep(0, length(distances))
      vars = rep(0, length(distances))
      for(iicv in 1:CV){
        # cat(sprintf("seed: %d, CV fold for c %d out of %d", seed, iicv, CV),file="results/outputs.txt",sep="\n",append=TRUE)
        # print(sprintf("seed: %d, CV fold for c %d out of %d", seed, icv, CV))
        validation_set = c(cutoffs[iicv]:cutoffs[iicv+1])
        n_train2 = n_train - length(validation_set)
        gender_data_cv = get_data(shuffled[-validation_set,])
        X_cv = gender_data[[1]]
        X_no_sex_cv = gender_data[[2]]
        y_var_cv = gender_data[[3]]
        fit_no_sex_cv = fit_model(X_no_sex_cv, y_var_cv)
        fit_no_sex_cv = fit_no_sex_cv[[1]]
        fit_sex_cv = fit_model(X_cv, y_var_cv)
        fit_sex_cv = fit_sex_cv[[1]]
        train_losses = get_loss(fit_sex_cv, fit_no_sex_cv, shuffled[validation_set,], distances, seed, J)
        losses = losses + colSums(train_losses)/n_train
        vars = vars + apply(train_losses, 2, var)/CV
      }
      if(loss_cache){
        save(losses, file = loss_file)
        save(vars, file = vars_file)
      }
    }
    chosen_distance = distances[which.min(losses)]
    losses = get_loss(fit_sex, fit_no_sex, df_validate, c(chosen_distance), seed, J)
  }
  return(list(losses, best_lam_no_sex, best_lam_sex, chosen_distance))
}



floodgate_CV <- function(int_df, J, CV, seed, distances, choose_dist, parallel = FALSE, loss_cache = FALSE){ 
  set.seed(seed)
  n = dim(int_df)[1]
  n_per_cv = floor(n/CV)
  cutoffs = c(1, n_per_cv * c(1:(CV-1)), n)
  shuffles = sample(1:n)
  shuffled = int_df[shuffles, ]
  if(choose_dist != 'all'){
    cv_mean = 0
    cv_var = 0
  } else{
    cv_mean = rep(0, length(distances))
    cv_var = rep(0, length(distances))
  }
  best_lam_sex = rep(0, CV)
  best_lam_no_sex = rep(0, CV)
  chosen_dists = rep(0, CV)
  start = Sys.time()
  for(icv in 1:CV){
    # cat(sprintf("seed: %f, CV fold %d out of %d", seed, icv, CV),file="results/outputs.txt",sep="\n",append=TRUE)
    elps = difftime(Sys.time(), start, unit = 'mins')[[1]]
    if(icv == 1)
      print(sprintf("seed: %d, CV fold %d out of %d", seed, icv, CV))
    if(icv > 1)
      print(sprintf("seed: %d, CV fold %d out of %d, time used: %.*f minutes; expected time to finish: %.*f minutes", seed, icv, CV, 2, elps, 2, (CV-icv+1)*elps/(icv-1)))
    validation_set = c(cutoffs[icv]:cutoffs[icv+1])
    res = floodgate(shuffled[-validation_set,], shuffled[validation_set,], J, seed, icv, CV, distances, choose_dist = choose_dist, parallel = parallel, loss_cache = loss_cache)
    best_lam_no_sex[icv] = res[[2]]
    best_lam_sex[icv] = res[[3]]
    chosen_dists[icv] = c(res[[4]])
    res = res[[1]]
    
    if(choose_dist != 'all'){
      cv_mean = cv_mean + sum(res)/n
      cv_var = cv_var + var(res)/CV
    } else{
      for(i in 1:length(distances)){
        cv_mean[i] = cv_mean[i] + sum(res[,i])/n
        cv_var[i] = cv_var[i] + var(res[,i])/CV
      }
    }
  }
  elps = difftime(Sys.time(), start, unit = 'mins')[[1]]
  # print(cv_mean)
  if(choose_dist != 'all'){
    csv = matrix(0, 1, 6+2+1)
    csv[1,] = c(median(chosen_dists), cv_mean, cv_var, 2*(1-cv_mean-1.644854*sqrt(cv_var/n)), 1-pnorm((1-cv_mean)/sqrt(cv_var/n)), seed, median(best_lam_no_sex), median(best_lam_sex), elps)
  } else{
    csv = matrix(0, length(distances), 6+2+1)
    for(i in 1:length(distances)){
      csv[i,] = c(distances[i], cv_mean[i], cv_var[i], 2*(1-cv_mean[i]-1.644854*sqrt(cv_var[i]/n)), 1-pnorm((1-cv_mean[i])/sqrt(cv_var[i]/n)), seed, median(best_lam_no_sex), median(best_lam_sex), elps)
    }
  }
  return(csv)
}