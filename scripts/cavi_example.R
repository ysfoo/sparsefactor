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

# run coordinate ascent variational inference (CAVI), using hyperparameters according to Section 5.1 of report
# 10 trials are run the trial with the largest converged ELBO is used for inference
# running all 10 trials takes ~2min on a 3.2 GHz CPU
save.freq <- 100 # how often are variational factors saved
best.elbo <- -Inf
for(seed in 21:30) {
    vi.res <- cavi(data$ymat, c(rep(0.1, S), rep(0.9, D)), 0.001, 0.001, 0.001, 0.001,
                   save=save.freq, max_iter=10000, seed=seed)

    n.iter <- tail(vi.res$iter, 1)
    idx <- n.iter / save.freq
    elbo <- vi.res$elbo[idx]
    elbo.change <- elbo - vi.res$elbo[idx-1]
    print(paste0('Trial ', seed-20, ': ', n.iter, ' iterations, ELBO = ', elbo))
    print(paste0('ELBO increased by ', elbo.change, ' in the last ', save.freq, ' iterations'))
    if(elbo > best.elbo) {
    	best.elbo <- elbo
    	best.seed <- seed
    }

    saveRDS(vi.res, paste0("../output/cavi_s", seed, ".rds"))
}
print(paste0('Trial with highest ELBO is trial ', best.seed-20))

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

# take best trial (largest converged ELBO)
vi.res <- readRDS(paste0("../output/cavi_s", best.seed, ".rds"))
n.iter <- tail(vi.res$iter, 1)
idx <- n.iter / save.freq

# posterior means
zmean <- vi.res$zmean[idx,,]
lmean <- vi.res$lmean[idx,,]
fmean <- vi.res$fmean[idx,,]
# posterior variance of acitvation matrix (F)
fvar <-  vi.res$fsig[idx,,]

# get relabelling to match posterior summary to simulated dataset
labelling <- get_relabelling(fmean, fvar, data$fmat)
# apply the relabelling
zmean.relabel <- zmean[,labelling$permutation]
lmean.relabel <- lmean[,labelling$permutation]*matrix(labelling$sign, nrow=G, ncol=K, byrow=TRUE)
fmean.relabel <- fmean[labelling$permutation,]*matrix(labelling$sign, nrow=K, ncol=N)

# calculate relative root mean square error of loading matrix (L) and activation matrix (F)
l.rrmse <- sqrt(sum((lmean.relabel-data$lmat)^2)/sum(data$lmat^2))
f.rrmse <- sqrt(sum((fmean.relabel-data$fmat)^2)/sum(data$fmat^2))

print(paste0("RRMSE of L: ", l.rrmse))
print(paste0("RRMSE of F: ", f.rrmse))

# inferred connectivity structure
png(paste0("../output/cavi_s", best.seed, "_heatmap.png"), width=3, height=5, units="in", res=1200)
print(levelplot(t(zmean.relabel),
				at=seq(0,1,0.05), aspect=2, col.regions=viridis(100),
				ylim=0.5+c(G,0), scales=list(y=list(at=c())),
				colorkey=list(width=1),
				main=list(label=paste0("VI posterior mean\n(trial ", best.seed-20, ")"), cex=1),
				xlab="Factors",
				ylab="800 features",
				par.settings=list(layout.heights=list(axis.top=0.5))))
dev.off()