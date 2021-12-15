library(sparsefactor)
library(viridis)
library(lattice)

# simulate data according to Section 5.1 of report
S <- 5 # sparse factors
D <- 1 # dense factors
K <- S + D
N <- 100
G <- 800

set.seed(2)
data <- sim.sfm(K=K, N=N, G=G,
				zmat = matrix(c(rep(0, 30 * G / 40), rep(1, 1 * G / 40), rep(0, 5 * G / 40), rep(1, 1 * G / 40), rep(0, 2 * G / 40),rep(1, 1 * G / 40),
							  rep(0, 6 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 1 * G / 40), rep(0, 1 * G / 40), rep(1, 1 * G / 40), rep(0, 13 * G / 40), rep(1, 3 * G / 40), rep(0, 1 * G / 40),
		                      rep(1, 1 * G / 40), rep(0, 4 * G / 40), rep(1, 1 * G / 40), rep(0, 14 * G / 40), rep(1, 2 * G / 40), rep(0, 8 * G / 40), rep(1, 6 * G / 40), rep(0, 4 * G / 40),
		                      rep(1, 5 * G / 40), rep(0, 15 * G / 40), rep(1, 10 * G / 40), rep(0, 10 * G / 40),
		                      rep(1, 20 * G / 40), rep(0, 20 * G / 40),
		                      rep(1, G)),
				nrow=G, ncol=K),
				alphavec=rep(1, K), snr=5)

# run Gibbs sampler, using hyperparameters according to Section 5.1 of report
# 5 chains of 2000 burn-in iterations, then another 3000*10 iterations
# but only every 10th iteration is kept
# running all 5 chains in succession takes ~1hr on a 3.2 GHz CPU
for(seed in 11:15) {
	samples <- gibbs(3000, data$ymat, c(rep(0.1, S), rep(0.9, D)),
					 0.001, 0.001, 0.001, 0.001,
					 thin=10, burn_in=2000, seed=seed)
    # undo sign-switching and label-switching within chain
    samples <- relabel_samples(samples)
    # save chain
    saveRDS(samples, paste0("../output/gibbs_s", seed, ".rds"))
}

# running some diagnostics that are across chains (e.g. R-hat or ESS) requires relabelling across chains
# use relabel_samples on a new list with all samples concatenated

# plot heatmap of simulated connectivity matrix
png("../output/true_heatmap.png", width=3, height=5, units="in", res=1200)
print(levelplot(t(data$zmat),
				at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
				ylim=0.5+c(G,0), scales=list(y=list(at=c())),
				colorkey=list(width=1),
				main=list(label="Connectivity matrix (truth)\n", cex=1),
				xlab="Factors",
				ylab="800 features",
				par.settings=list(layout.heights=list(axis.top=0.5))))
dev.off()

for(seed in 11:15) {
    samples <- readRDS(paste0("../output/gibbs_s", seed, ".rds"))

    # posterior means
	zmean <- colMeans(samples$zmat)
	lmean <- colMeans(samples$lmat)
	fmean <- colMeans(samples$fmat)
	# posterior variance of acitvation matrix (F)
	fvar <-  colMeans(samples$fmat*samples$fmat)-fmean*fmean

	# get relabelling to match posterior summary to simulated dataset
	labelling <- get_relabelling(fmean, fvar, data$fmat)
	# apply the relabelling
	zmean.relabel <- zmean[,labelling$permutation]
	lmean.relabel <- lmean[,labelling$permutation]*matrix(labelling$sign, nrow=G, ncol=K, byrow=TRUE)
	fmean.relabel <- fmean[labelling$permutation,]*matrix(labelling$sign, nrow=K, ncol=N)

	# calculate relative root mean square error of loading matrix (L) and activation matrix (F)
	l.rrmse <- sqrt(sum((lmean.relabel-data$lmat)^2)/sum(data$lmat^2))
	f.rrmse <- sqrt(sum((fmean.relabel-data$fmat)^2)/sum(data$fmat^2))

	print(paste0("Chain ", seed-10))
	print(paste0("RRMSE of L: ", l.rrmse))
	print(paste0("RRMSE of F: ", f.rrmse))


	# inferred connectivity structure
	png(paste0("../output/gibbs_s", seed, "_heatmap.png"), width=3, height=5, units="in", res=1200)
	print(levelplot(t(zmean.relabel),
					at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
					ylim=0.5+c(G,0), scales=list(y=list(at=c())),
					colorkey=list(width=1),
					main=list(label=paste0("MCMC posterior mean\n(chain ", seed-10, ")"), cex=1),
					xlab="Factors",
					ylab="800 features",
					par.settings=list(layout.heights=list(axis.top=0.5))))
	dev.off()
}