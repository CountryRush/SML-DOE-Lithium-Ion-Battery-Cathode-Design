# ============================================================
# R analysis for proposal-aligned 50-run dataset
# 2^3 factorial (2 blocks, ABC confounded) + LHS + CCD
# Outputs: results/ (tables + figures)
# ============================================================
dir.create("results/figures", recursive = TRUE, showWarnings = FALSE)

dat <- read.csv("data/proposal_doe_dataset.csv", stringsAsFactors = FALSE)
dat$combo <- trimws(dat$combo)

# ---- derive coded factors from combo ----
code <- list(
  "(1)" = c(-1,-1,-1), "a"  = c(1,-1,-1), "b" = c(-1,1,-1), "ab" = c(1,1,-1),
  "c"   = c(-1,-1,1),  "ac" = c(1,-1,1),  "bc"= c(-1,1,1),  "abc"= c(1,1,1)
)
f <- subset(dat, design == "2^3")
f$A <- sapply(f$combo, function(k) code[[k]][1])
f$B <- sapply(f$combo, function(k) code[[k]][2])
f$C <- sapply(f$combo, function(k) code[[k]][3])
f$block <- factor(f$block)

# ---- 1. Treatment means ----
means <- aggregate(specific_capacity_mAh_g ~ combo + A + B + C, data = f,
                   FUN = function(x) c(mean = mean(x), sd = sd(x)))
means_df <- data.frame(combo = means$combo, A = means$A, B = means$B, C = means$C,
                       mean = round(means$specific_capacity_mAh_g[, "mean"], 3),
                       sd = round(means$specific_capacity_mAh_g[, "sd"], 3))
write.csv(means_df, "results/treatment_means.csv", row.names = FALSE)

# ---- 2. Effects (2*beta) with 95% CI ----
m_full <- lm(specific_capacity_mAh_g ~ A*B*C, data = f)
cf <- summary(m_full)$coefficients
terms <- c("A","B","C","A:B","A:C","B:C","A:B:C")
eff <- data.frame(
  term = terms,
  estimate = 2 * cf[terms, "Estimate"],
  se = 2 * cf[terms, "Std. Error"],
  t = cf[terms, "t value"],
  p = cf[terms, "Pr(>|t|)"]
)
eff$ci_low  <- eff$estimate - qt(0.975, df = m_full$df.residual) * eff$se
eff$ci_high <- eff$estimate + qt(0.975, df = m_full$df.residual) * eff$se
eff$estimate <- round(eff$estimate, 3)
eff$se <- round(eff$se, 3)
eff$ci_low <- round(eff$ci_low, 3)
eff$ci_high <- round(eff$ci_high, 3)
eff$p <- signif(eff$p, 4)
write.csv(eff, "results/effects.csv", row.names = FALSE)

# ---- 3. ANOVA (block + main + 2FI; ABC aliased with block) ----
m_anova <- lm(specific_capacity_mAh_g ~ block + A + B + C + A:B + A:C + B:C, data = f)
sink("results/anova.txt")
cat("ANOVA: specific_capacity_mAh_g ~ block + A + B + C + A:B + A:C + B:C\n")
cat("Note: ABC is confounded with block (block contrast = ABC contrast).\n\n")
print(anova(m_anova))
cat("\nResidual df:", df.residual(m_anova), "\n")
cat("Pure error SD:", round(sigma(m_anova), 4), "mAh/g\n")
sink()
# restore console output
cat("ANOVA saved.\n")

# ---- 4. Assumption checks ----
sw <- shapiro.test(resid(m_anova))
sink("results/assumptions.txt")
cat("Shapiro-Wilk normality test on residuals\n")
print(sw)
cat("\nResidual SD:", round(sigma(m_anova), 4), "\n")
sink()

# ---- 5. Figures ----
png("results/figures/fig_main_effects.png", width = 1100, height = 750)
par(mfrow = c(1,3), mar = c(4,4,3,1))
for (nm in c("A","B","C")) {
  m <- tapply(f$specific_capacity_mAh_g, f[[nm]], mean)
  plot(c(1,2), m, type = "b", pch = 19, xaxt = "n", xlab = nm, ylab = "Mean capacity (mAh/g)",
       main = paste("Main effect", nm), xlim = c(0.8, 2.2))
  axis(1, at = c(1,2), labels = c("-1", "+1"))
  grid()
}
dev.off()

png("results/figures/fig_interaction_AB.png", width = 800, height = 600)
interaction.plot(f$A, f$B, f$specific_capacity_mAh_g, xlab = "A (thickness)", ylab = "Mean capacity (mAh/g)",
                 trace.label = "B (porosity)", main = "Interaction A x B")
dev.off()

png("results/figures/fig_interaction_AC.png", width = 800, height = 600)
interaction.plot(f$A, f$C, f$specific_capacity_mAh_g, xlab = "A (thickness)", ylab = "Mean capacity (mAh/g)",
                 trace.label = "C (radius)", main = "Interaction A x C")
dev.off()

png("results/figures/fig_interaction_BC.png", width = 800, height = 600)
interaction.plot(f$B, f$C, f$specific_capacity_mAh_g, xlab = "B (porosity)", ylab = "Mean capacity (mAh/g)",
                 trace.label = "C (radius)", main = "Interaction B x C")
dev.off()

png("results/figures/fig_qq.png", width = 800, height = 600)
qqnorm(resid(m_anova), pch = 19, main = "Normal Q-Q (residuals)")
qqline(resid(m_anova), col = "red")
dev.off()

png("results/figures/fig_resid_fitted.png", width = 800, height = 600)
plot(fitted(m_anova), resid(m_anova), pch = 19, xlab = "Fitted (mAh/g)", ylab = "Residuals",
     main = "Residuals vs Fitted")
abline(h = 0, lty = 2, col = "gray")
dev.off()

# ---- 6. GP surrogate on all 50 points (DiceKriging) ----
suppressPackageStartupMessages(library(DiceKriging))
g <- dat
X <- data.frame(
  A = (g$A_thickness_um - 40) / (200 - 40),
  B = (g$B_porosity - 0.20) / (0.60 - 0.20),
  C = (g$C_radius_um - 1) / (12 - 1)
)
y <- g$specific_capacity_mAh_g
km_fit <- km(~A+B+C, design = X, response = y, covtype = "matern5_2",
             nugget.estim = TRUE, control = list(trace = FALSE))
loo <- leaveOneOut.km(km_fit, type = "UK")
rmse <- sqrt(mean((loo$mean - y)^2))
r2 <- 1 - sum((loo$mean - y)^2) / sum((y - mean(y))^2)
sink("results/gp_summary.txt")
cat("GP (Matern 5/2 + linear trend), n =", nrow(X), "\n")
cat("LOO-RMSE:", round(rmse, 4), "mAh/g\n")
cat("LOO-R2:", round(r2, 4), "\n\n")
print(km_fit)
sink()

cat("All done. Results in results/.\n")
