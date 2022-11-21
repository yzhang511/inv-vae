library(reticulate)
np <- import("numpy")

# devtools::install_github("emanuelealiverti/SOG")
# require(sOG)

# orthogonal-to-group decomposition
OG = function(
  X, 
  z, 
  K = max(2, round(NCOL(X)/10)), 
  rescale = T) {
    if(!is.matrix(X)) stop("X must be a matrix")
    if(rescale) X = scale(X)
    SVD = svd(X, nu = K, nv = K)
    temp = lm(SVD$u %*% diag(SVD$d[1:K]) ~ z)
    S = SVD$u %*% diag(SVD$d[1:K]) - cbind(1,z)%*%temp$coef
    return(list(S = S, U = t(SVD$v)))
}

# load data
brain_networks = np$load("./reproducible_results/saved_results/hcp_brain_networks.npy")
traits_lst = np$load("./reproducible_results/saved_results/hcp_traits.npy")
motion = np$load("./reproducible_results/saved_results/hcp_motion.npy")

# 5-fold CV

nfolds = 5
K = 68
cv_rmses = vector(length = nfolds)
seeds = 1:nfolds

traits = traits_lst[,5]

for(i in 1:nfolds) {
  print(i)
  seed = seeds[i]
  set.seed(seed)
  n = dim(brain_networks)[1]
  
  train_ids = sample(1:n, .8*n)
  test_ids = setdiff(1:n, train_ids)
  
  require(Matrix)
    X = scale(brain_networks)
    
    for(j in 1:ncol(X)){
      X[is.na(X[,j]), j] = 0
  }
  
  res_fpl = OG(X = X[train_ids,], 
               z = motion[train_ids], 
               K = K, 
               rescale = F)
  
  y_train = traits[train_ids]
  y_test = traits[test_ids]
  
  lr = lm(y_train ~ res_fpl$S)
  
  res_fpl_test = OG(X = X[y_test,], 
                    z = motion[y_test], 
                    K = K, 
                    rescale = F)
  
  pred = list(
    "OG" = cbind(1, res_fpl_test$S) %*% coef(lr)
  )
  
  motion_test = (motion[test_ids])[!is.na(y_test)]
  y_pred = as.numeric(pred$OG)[!is.na(y_test)]
  y_test = y_test[!is.na(y_test)]
  
  cv_rmses[i] = sqrt(mean((y_pred - y_test)^2))
}

mean(cv_rmses)
print(cv_rmses)
