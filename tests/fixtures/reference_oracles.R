# Independent base-R numerical oracles. No packages installed, no files deleted.
# Arguments: fixture directory, author's pinned MICoDa_Aux.R, output directory.
args <- commandArgs(trailingOnly=TRUE)
fixture <- args[1]; author_source <- args[2]; output <- args[3]
if (dir.exists(output)) stop('Refusing to overwrite oracle output directory')
dir.create(output, recursive=TRUE)
source(author_source)
data <- read.csv(file.path(fixture, 'micoda_input.csv'))
for (v in c(TRUE, FALSE)) {
  result <- MICoDa(data, D=4, varlist='covariate', iter=5, rescaleAll=v,
                  Constant=1, ZeroAsMissing=FALSE)
  write.csv(result$data_Imp[,1:4],file.path(output,paste0('micoda_',if(v)'v1' else 'v2','.csv')),row.names=FALSE)
}
clr_oracle <- function(x) log(x) - rowMeans(log(x))
knn <- function(x, donors, k=3, adj='median', complete=FALSE) {
  answer <- x
  for (i in 1:nrow(x)) {
    observed <- which(!is.na(x[i,]))
    for (j in which(is.na(x[i,]))) {
      eligible <- if(complete) which(complete.cases(donors)) else which(complete.cases(donors[,unique(c(observed,j)),drop=FALSE]))
      distances <- sqrt(rowSums((clr_oracle(donors[eligible,observed,drop=FALSE])-
                    matrix(clr_oracle(x[i,observed,drop=FALSE]),nrow=length(eligible),ncol=length(observed),byrow=TRUE))^2))
      selected <- eligible[order(distances,eligible)][seq_len(min(k,length(eligible)))]
      reduce <- if(adj=='median') median else sum
      adjustment <- reduce(x[i,observed])/apply(donors[selected,observed,drop=FALSE],1,reduce)
      answer[i,j] <- median(adjustment*donors[selected,j])
    }
  }
  answer
}
x <- as.matrix(read.csv(file.path(fixture,'hron_input.csv')))
write.csv(knn(x,x),file.path(output,'hron_knn.csv'),row.names=FALSE)
basis <- function(d) {
  b <- matrix(0,d,d-1)
  for(j in seq_len(d-1)) {remaining=d-j; b[j,j]=-sqrt(remaining/(remaining+1));b[(j+1):d,j]=1/sqrt(remaining*(remaining+1))}
  b
}
state <- knn(x,x); observed <- !is.na(x); d <- ncol(x)
ordering <- order(-colSums(is.na(x)),seq_len(d)); b <- basis(d)
for(iteration in 1:3) {
  for(target in ordering) {
    permutation <- c(target,ordering[ordering!=target]); z <- clr_oracle(state[,permutation])%*%b
    design <- cbind(1,z[,-1,drop=FALSE]); rows <- observed[,target]
    beta <- qr.solve(design[rows,,drop=FALSE],z[rows,1])
    missing <- !rows
    if(any(missing)) {
      z[missing,1] <- design[missing,,drop=FALSE]%*%beta
      reconstructed <- exp(z%*%t(b)); reconstructed <- reconstructed/rowSums(reconstructed)
      state[,permutation] <- reconstructed
    }
  }
}
write.csv(state,file.path(output,'hron_ls.csv'),row.names=FALSE)

# Exact raw LTS objective using exhaustive h-subsets on a small known problem.
design <- cbind(1,0:6); y <- 2+3*(0:6);y[7] <- y[7]+100;h <- 5
best <- Inf; beta_best <- NULL
for(s in combn(1:7,h,simplify=FALSE)) {
 beta <- qr.solve(design[s,],y[s]);objective <- sum(sort((y-design%*%beta)^2)[1:h])
 if(objective<best) {best=objective;beta_best=beta}
}
write.csv(matrix(beta_best,nrow=1),file.path(output,'lts.csv'),row.names=FALSE)

# Alpha-kNN independently from equations, full donor mean before subset closure.
donors <- as.matrix(read.csv(file.path(fixture,'tsagris_donors.csv')))
query <- as.matrix(read.csv(file.path(fixture,'tsagris_query.csv')))
jsd_oracle <- function(p,q) {
 middle <- (p+q)/2
 a <- ifelse(p==0,0,p*log(p/middle));b <- ifelse(q==0,0,q*log(q/middle));sum(a+b)
}
for(alpha in c(-.5,0,.5,1)) {
 answer <- query
 for(i in 1:nrow(query)) {
  known <- which(!is.na(query[i,]));missing <- which(is.na(query[i,]))
  q <- query[i,known]/sum(query[i,known]);sub <- donors[,known,drop=FALSE];sub<-sub/rowSums(sub)
  dist <- apply(sub,1,function(p)jsd_oracle(p,q));chosen<-donors[order(dist)[1:3],,drop=FALSE]
  if(alpha==0) {mu<-exp(colMeans(log(chosen)))} else {powers<-chosen^alpha;powers<-powers/rowSums(powers);mu<-colMeans(powers)^(1/alpha)}
  mu<-mu/sum(mu);answer[i,missing]<-(1-sum(query[i,known]))*mu[missing]/sum(mu[missing])
 }
 write.csv(answer,file.path(output,paste0('tsagris_',alpha,'.csv')),row.names=FALSE)
}
cat('Base-R source/equation oracles completed\n')
