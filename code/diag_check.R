
dat <- read.csv("data/proposal_doe_dataset.csv", stringsAsFactors = FALSE)
dat$combo <- trimws(dat$combo)
code <- list("(1)"=c(-1,-1,-1),"a"=c(1,-1,-1),"b"=c(-1,1,-1),"ab"=c(1,1,-1),
             "c"=c(-1,-1,1),"ac"=c(1,-1,1),"bc"=c(-1,1,1),"abc"=c(1,1,1))
f <- subset(dat, design=="2^3")
f$A <- sapply(f$combo, function(k) code[[k]][1])
f$B <- sapply(f$combo, function(k) code[[k]][2])
f$C <- sapply(f$combo, function(k) code[[k]][3])
f$block <- factor(f$block)
m <- lm(specific_capacity_mAh_g ~ block + A + B + C + A:B + A:C + B:C, data=f)
r <- resid(m)
out <- data.frame(run=f$run, combo=f$combo, block=f$block, y=f$specific_capacity_mAh_g,
                  fitted=fitted(m), resid=round(r,3))
out <- out[order(-abs(r)), ]
print(out, row.names=FALSE)
cat("\nShapiro p:", shapiro.test(r)$p.value, "\n")
cat("Residual SD:", sd(r), "\n")
cat("Max |resid|:", max(abs(r)), "\n")
