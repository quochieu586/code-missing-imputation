source('tests/fixtures/impKNNa.R')

cenLR <- function(x) {
  if (is.vector(x)) x <- matrix(x, nrow=1)
  gm <- exp(rowMeans(log(x)))
  list(x.clr = log(x / gm))
}

df <- read.csv('tests/fixtures/expenditures.csv')
x <- as.matrix(df)
x[1, 3] <- NA

cat("Testing impKNNa from R:\n")
for (k_val in 1:6) {
  res <- impKNNa(x, k=k_val, metric="Aitchison", adj="median", agg="median")
  cat(sprintf("  k=%d: imputed = %.2f\n", k_val, res$xImp[1, 3]))
}

cat("\nTesting with das=TRUE (formula 2 definition of Aitchison distance):\n")
for (k_val in 1:6) {
  res <- impKNNa(x, k=k_val, metric="Aitchison", das=TRUE, adj="median", agg="median")
  cat(sprintf("  k=%d (das=TRUE): imputed = %.2f\n", k_val, res$xImp[1, 3]))
}
