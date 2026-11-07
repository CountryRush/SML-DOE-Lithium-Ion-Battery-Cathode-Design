# ============================================================
# Bayesian optimization comparison (R)
# Black box = GP surrogate fitted on the 50-run dataset
# Random vs BO-EI vs BO-UCB, 8 initial LHS + 20 iterations, 3 seeds
# ============================================================
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(DiceKriging))

dat <- read.csv("data/proposal_doe_dataset.csv", stringsAsFactors = FALSE)
Xall <- data.frame(
  A = (dat$A_thickness_um - 40) / (200 - 40),
  B = (dat$B_porosity - 0.20) / (0.60 - 0.20),
  C = (dat$C_radius_um - 1) / (12 - 1)
)
yall <- dat$specific_capacity_mAh_g

# "truth" surrogate
truth <- km(~A+B+C, design = Xall, response = yall, covtype = "matern5_2",
            nugget.estim = TRUE, control = list(trace = FALSE))
eval_box <- function(X) predict(truth, newdata = X, type = "UK")$mean

# initial LHS (simple random stratified approximation)
lhs_init <- function(n, seed) {
  set.seed(seed)
  X <- matrix(NA, n, 3)
  for (j in 1:3) {
    u <- (sample(0:(n-1)) + runif(n)) / n   # stratified
    X[, j] <- u
  }
  out <- as.data.frame(X)
  names(out) <- c("A","B","C")
  out
}

run_random <- function(seed) {
  set.seed(seed)
  X <- as.data.frame(matrix(runif(28*3), 28, 3)); names(X) <- c("A","B","C")
  y <- eval_box(X)
  cummax(y)
}

run_bo <- function(seed, acq, initX) {
  set.seed(seed)
  y <- eval_box(initX)
  best <- cummax(y)
  X <- initX
  for (t in 1:20) {
    m <- km(~A+B+C, design = X, response = y, covtype = "matern5_2",
            nugget.estim = TRUE, control = list(trace = FALSE))
    cand <- as.data.frame(matrix(runif(2000*3), 2000, 3)); names(cand) <- c("A","B","C")
    pr <- predict(m, newdata = cand, type = "UK", se.compute = TRUE)
    mu <- pr$mean; sdv <- pmax(pr$sd, 1e-9)
    ybest <- max(y)
    if (acq == "EI") {
      z <- (mu - ybest - 0.01) / sdv
      a <- (mu - ybest - 0.01) * pnorm(z) + sdv * dnorm(z)
    } else {
      a <- mu + 2.0 * sdv
    }
    xnew <- cand[which.max(a), , drop = FALSE]
    ynew <- eval_box(xnew)
    X <- rbind(X, xnew); y <- c(y, ynew)
    best <- c(best, max(best[length(best)], ynew))
  }
  best
}

seeds <- c(1,2,3)
R <- sapply(seeds, run_random)
E <- sapply(seeds, function(s) run_bo(s, "EI", lhs_init(8, s)))
U <- sapply(seeds, function(s) run_bo(s, "UCB", lhs_init(8, s)))
Rm <- rowMeans(R); Em <- rowMeans(E); Um <- rowMeans(U)

target <- 300
ev <- function(v) { i <- which(v >= target); if (length(i)) i[1] else NA }
res <- data.frame(
  method = c("Random","BO-EI","BO-UCB"),
  final_best = round(c(Rm[28], Em[28], Um[28]), 3),
  evals_to_target = c(ev(Rm), ev(Em), ev(Um))
)
write.csv(res, "results/bo_results.csv", row.names = FALSE)

png("results/figures/fig_bo.png", width = 900, height = 650)
plot(1:28, Rm, type = "o", pch = 19, ylim = range(c(Rm,Em,Um)),
     xlab = "Evaluations", ylab = "Best capacity found (mAh/g)",
     main = "Bayesian optimization: equal-budget comparison")
lines(1:28, Em, type = "o", pch = 17, col = "blue")
lines(1:28, Um, type = "o", pch = 15, col = "darkgreen")
abline(h = target, lty = 2, col = "red")
legend("bottomright", legend = c("Random","BO-EI","BO-UCB"),
       pch = c(19,17,15), col = c("black","blue","darkgreen"), lty = 1)
dev.off()

sink("results/bo_summary.txt")
cat("BO comparison (3 seeds), black box = GP surrogate of 50-run data\n")
cat("Target:", target, "mAh/g\n\n")
print(res)
cat("\nRandom final best:", round(Rm[28],2), "\n")
cat("BO-EI final best :", round(Em[28],2), "\n")
cat("BO-UCB final best:", round(Um[28],2), "\n")
sink()
cat("BO done.\n")
