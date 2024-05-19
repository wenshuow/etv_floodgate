# Replication Code for "Using Machine Learning to Test Causal Hypotheses in Conjoint Analysis" by Ham, Imai, and Janson. (2022)
# source code for obtaining all HierNet related test statistics 

library(hierNet)
# library(car)

#Helper function to obtain cross-validated lambda using HierNet
# We output the best lambda using the best MSE prediction (predicting based on probabilities)
get_lam = function(lambda, nfolds = 3, X, y_var, seed = sample(1:1000, size = 1), unconstrained = FALSE) {
  # If we do not impose the no profile order effect (unconstrained = TRUE) then the cross validation split can be done at random
  # If we do impose the no profile order effect we must respect our augmented data structure when performing the train/test split
  if (unconstrained == TRUE) {
    half = nrow(X)
    set.seed(seed)
    random_idx = split(sample(half, half, replace = FALSE), as.factor(1:nfolds))
  } else {
    half = (nrow(X)/2)
    set.seed(seed)
    random_idx = split(sample(half, half, replace = FALSE), as.factor(1:nfolds))
    for (i in 1:nfolds) {
      random_idx[[i]] = c(random_idx[[i]], random_idx[[i]] + half)
    }
  }
  
  
  error_list_prob = list()
  
  for (i in 1:nfolds) {
    errors_prob = vector()
    test_idx = random_idx[[i]]
    train_idx = (1:nrow(X))[-test_idx]
    
    invisible(capture.output(cv_hiernets <- hierNet.logistic.path(as.matrix(X)[train_idx, ], y_var[train_idx],
                                                                  lamlist = lambda, diagonal = FALSE)))
    predicted_y_prob = predict(cv_hiernets, X[test_idx, ])$prob
    for (j in 1:length(lambda)) {
      errors_prob[j] = mean((y_var[test_idx] - predicted_y_prob[,j])^2)
    }
    error_list_prob[[i]] = errors_prob
  }
  
  
  cv_errors_prob = vector()
  for (i in 1:length(lambda)) {
    cv_errors_prob[i] = mean(sapply(error_list_prob, "[[", i))
  }
  
  gotten_lam = lambda[which.min(cv_errors_prob)]
  return(gotten_lam)
}

# All HierNet Functions below take input: 
# hiernet_object: A class of HierNet fit (from HierNet package)   
# idx: Indexes indicating which relevant factor levels of a given matrix X we want to sum from all coefficients
# X: Model matrix of X that is inputted to hiernet_object fit

# Main HierNet test statistic as implemented in Equation 5 and 11
# Additional inputs: 
# Analysis: if TRUE function also returns which interactions are strongest (heuristically viewing which interactions contributed most to observed test statistic)
# Forced: if non-null takes input of which indexes of X matrix we wish to "force" as main effect (see Section 5.2 for further details)
# Group: list (total length of unlisted list should match idx) that "groups" up all relevant interested indexes in idx to implement Equation 5 (only necessary when idx contains indexes we do not want to average the respective effects for and compare together as done in Equation 5)
# Immigration Example: idx = indexes of matrix X that refer to Mexico or European (length of 2)
# Gender Example: idx = indexes of matrix X that refers to Male or Female and all forced Male/Female interactions with Party
# Gender Example: forced = indexes of matrix X of all forced interactions with party affiliation and party affiliation main effects
# Gender Example: group = list(c(1,2), c(3, 4), c(5, 6), c(7,8), c(9, 10)), where c(1,2) for example compares all main and interaction of just male and female. c(3,4) compares all main and interaction effects of male and one factor level from party affiliation, etc. 
hiernet_group = function(hiernet_object, idx, X, analysis = TRUE, forced = NULL, group = NULL) {
  # No grouping case
  if (is.null(group)) {
    # Sums main effects
    main = hiernet_object$bp[idx] - hiernet_object$bn[idx]
    main_means = mean(main)
    main_diff = sum((main - main_means)^2)
    
    #all interaction effects
    if (is.null(forced)) {
      I_int = list()
      for (i in 1:length(idx)) {
        I_int[[i]] = (hiernet_object$th[idx[i], ] + hiernet_object$th[, idx[i]])/2
      }
      
      int_means = Reduce("+", I_int)/length(idx)
      int_diff = vector()
      for (i in 1:length(idx)) {
        int_diff[i] = sum((I_int[[i]] - int_means)^2)
      }
    } else {
      # if there are forced interactions we take them out to avoid double counting
      I_int = list()
      for (i in 1:length(idx)) {
        I_int[[i]] = (hiernet_object$th[idx[i], -forced] + hiernet_object$th[-forced, idx[i]])/2
      }
      
      int_means = Reduce("+", I_int)/length(idx)
      int_diff = vector()
      for (i in 1:length(idx)) {
        int_diff[i] = sum((I_int[[i]] - int_means)^2) # we use the fact that we only need take one side of difference since we have two coefficients to compare
      }
      
    }
    
    ts = main_diff + sum(int_diff) 
    
    # TRUE only when computing observed test statistics for main application analysis
    if (analysis == TRUE) {
      main_contribution = main_diff
      int_contribution = sum(int_diff)
      individual_ints = (I_int[[1]] - int_means)^2
      influential_factors = colnames(X)[order(individual_ints, decreasing = TRUE)][1:2]
      contribution = individual_ints[order(individual_ints, decreasing = TRUE)][1:2]
      
      analysis_df = data.frame(ts, main_contribution, int_contribution, influential_factor_1 = influential_factors[1], influential_factor_2 = influential_factors[2], contribution_1 = contribution[1], contribution_2 = contribution[2])
      
      return(analysis_df)    
      
    } else {
      return(ts)
    }
  } else {
    # if there are groupings we sum and average within each group (same for interactions below)
    main = hiernet_object$bp[idx] - hiernet_object$bn[idx]
    main_contribution = vector()
    for (i in 1:length(group)) {
      main_contribution[i] = sum((main[group[[i]]] - mean(main[group[[i]]]))^2)
    }
    
    if (is.null(forced)) {
      I_int = list()
      for (i in 1:length(idx)) {
        I_int[[i]] = (hiernet_object$th[idx[i], ] + hiernet_object$th[, idx[i]])/2
      }
    } else{
      I_int = list()
      for (i in 1:length(idx)) {
        I_int[[i]] = (hiernet_object$th[idx[i], -forced] + hiernet_object$th[-forced, idx[i]])/2
      }
    }
    
    
    int_contribution = vector()
    tracking_individual_int = list()
    for (i in 1:length(group)) {
      in_int = I_int[group[[i]]]
      int_means = Reduce("+", in_int)/length(in_int)
      tracking_individual_int[[i]] = (in_int[[1]] - int_means)^2
      contribution = vector()
      for (j in 1:length(in_int)) {
        contribution[j] = sum((in_int[[j]] - int_means)^2)
      }
      int_contribution[i] = sum(contribution)
    }
    
    ts = sum(main_contribution) + sum(int_contribution) 
    
    if (analysis == TRUE) {
      main_contribution = sum(main_contribution)
      int_contribution = sum(int_contribution)
      largest_ints = largest_ints_idx = vector()
      for (i in 1:length(group)) {
        largest_ints[i] = max(tracking_individual_int[[i]])
        largest_ints_idx[i] = which.max(tracking_individual_int[[i]])
        
      }
      largest_group = which.max(largest_ints)
      second_largest_group = order(largest_ints, decreasing = TRUE)[2]
      
      largest_int_contributer = colnames(X)[-forced][largest_ints_idx[largest_group]]
      second_largest_int_contributer = colnames(X)[-forced][largest_ints_idx[second_largest_group]]
      
      largest_interaction_with = colnames(X)[idx[group[[largest_group]]]][1]
      second_largest_interaction_with = colnames(X)[idx[group[[second_largest_group]]]][1]
      analysis_df = data.frame(ts, main_contribution, int_contribution, largest_int_contributer, largest_interaction_with, second_largest_int_contributer, second_largest_interaction_with)
      
      return(analysis_df)    
      
    } else {
      return(ts)
    }
  }
}


# Implementing Test statistic to test no profile order effect (See Section 3.5 under 'Profile Order Effect')
# idx_1: All indexes of matrix X corresponding to the left profile's factor levels
# idx_2: The same for the right profile's factor levels
# respond_idx: Indexes of the respondent characteristics
PO_stat = function(hiernet_object, idx_1, idx_2, respond_idx) {
  #main effects
  main_1 = hiernet_object$bp[idx_1] - hiernet_object$bn[idx_1]
  main_2 = hiernet_object$bp[idx_2] - hiernet_object$bn[idx_2]
  
  #within profile interactions
  within_int_list_left = list()
  within_int_list_right = list()
  
  for (i in 1:length(idx_1)) {
    within_int_list_left[[i]] = (hiernet_object$th[idx_1[i], idx_1[-i]] + hiernet_object$th[idx_1[-i], idx_1[i]])/2
    within_int_list_right[[i]] = (hiernet_object$th[idx_2[i], idx_2[-i]] + hiernet_object$th[idx_2[-i], idx_2[i]])/2
  }
  
  within_diff = unlist(Map("+", within_int_list_left, within_int_list_right))
  
  #between profile interactions
  between_int_list_left = list()
  between_int_list_right = list()
  
  for (i in 1:length(idx_1)) {
    between_int_list_left[[i]] = (hiernet_object$th[idx_1[i], idx_2] + hiernet_object$th[idx_2, idx_1[i]])/2
    between_int_list_right[[i]] = (hiernet_object$th[idx_2[i], idx_1] + hiernet_object$th[idx_1, idx_2[i]])/2
  }
  
  between_diff = unlist(Map("+", between_int_list_left, between_int_list_right))
  
  
  #respondent interactions
  R_int_list_left = list()
  R_int_list_right = list()
  
  for (i in 1:length(idx_1)) {
    R_int_list_left[[i]] = (hiernet_object$th[idx_1[i], respond_idx] + hiernet_object$th[respond_idx, idx_1[i]])/2
    R_int_list_right[[i]] = (hiernet_object$th[idx_2[i], respond_idx] + hiernet_object$th[respond_idx, idx_2[i]])/2
  }
  
  respondent_effects = unlist(Map("+", R_int_list_left, R_int_list_right))
  
  #division of two in the interactions because I overcount 
  stat= sum((main_1 + main_2)^2) + sum(within_diff^2/2) + sum(between_diff^2)/2 + sum((respondent_effects)^2)
  return(stat)
}

# Implementing Test statistic to test carryover effect(See Section 3.5 under 'Carryover Effect')
CO_stat = function(hiernet_object, idx) {
  I_list = list()
  for (i in 1:length(idx)) {
    I_list[[i]] = (hiernet_object$th[idx[i], -idx] + hiernet_object$th[-idx, idx[i]])/2
  }
  ts = sum(unlist(I_list)^2)/4
  return(ts)
}



