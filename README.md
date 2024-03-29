# sparsefactor: Bayesian inference techniques for sparse factor models

This project provides two approaches to inference for sparse factor models: Markov chain Monte Carlo (MCMC) and variational inference (VI). Additionally, a relabelling algorithm is implemented to address issues which arise from label switching and sign switching.

The GTEx eQTL summary data is provided as `gtexEQTL_zscore.rds` in the root directory.

## Background

In dimension reduction, *factor analysis* is an approach that discovers latent variables (factors) that explain the data. Given a large number of observations, each consisting of a large number of features, factor analysis seeks to quantify the associations between these features and some factors, and also the weight of each factor for each observation. Dimension reduction is achieved by having a number of factors that is far smaller than the number of features/observations.

If it is desired for some factors to govern only a small subset of features, the approach is then known as *sparse factor analysis*. This is because the matrix connecting the features to the factors will contain a significant proportion of zeros, making this loading matrix a sparse matrix. This is desirable as the resulting factors are easier to interpret.

One specific application of sparse factor models is the analysis of gene expression data. Biological theory expects that gene expression levels to be regulated by different mechanisms, which may be difficult to observe directly. Sparse factor analysis is an appropriate approach to analyse these processes, as most mechanisms (e.g. transcription factors) are each responsible for regulating only a small subset of genes. Successful inference may then shed light on the underlying gene regulatory network.

 The gene expression level of gene *i* of individual *j* can be modelled as

 <p align="center"><img src="https://render.githubusercontent.com/render/math?math=y_{ij} = \sum_{k=1}^K l_{ik}f_{kj}%2Be_{ij},"></p>

where *l*<sub><i>ik</i></sub> is the regulation strength of factor *k* on gene *i*, *f*<sub><i>kj</i></sub> is the activation weight of factor *k* for individual *j*, and *e*<sub><i>ij</i></sub> is a noise term. To achieve sparsity, *l*<sub><i>ik</i></sub> should be zero if gene *i* is not regulated by factor *k*. Further detail can be found in `docs/assets/report.pdf`.

Following a Bayesian approach, the posterior distributions of the parameters of interest (e.g. *l*<sub><i>ik</i></sub> and *f*<sub><i>kj</i></sub>) are intractable to calculate directly. Two techniques are implemented:

1. **Markov chain Monte Carlo (MCMC).** The posterior distribution is approximated by simulating samples from a Markov chain which converges to the desired distribution. When successful, the target distribution will be accurately caputred, but this approach is known to be computationally intensive. Specifically, I implemented a Gibbs sampler.

2. **Variational inference (VI).** As the posterior distribution involves complex dependencies between the model parameters, we instead search for an approximate distribution from a family of much simpler distributions. Solving this optimisation problem is computationally more efficient than MCMC, at the cost of fidelity due to the use of simpler distributions. Specifically, I used an mean-field approximation and implemented coordinate ascent variational inference (CAVI).

It is expected that VI should have an advantage of being the faster approach without sacrificing too much accuracy.

## Usage

Please ensure that [GNU Scientific Library (GSL)](https://www.gnu.org/software/gsl/) is installed. The functionality can be accessed by building an `R` package:

   ```R
   install.packages("devtools")
   library(devtools)
   install_github("ysfoo/sparsefactor")
   library(sparsefactor)
   ```

Some known errors:

- There is a known error involving `-lgfortran` or `-lquadmath` with installing `RcppArmadillo` (a dependency of the package) on macOS. Please follow the solution [here](https://thecoatlessprofessor.com/programming/cpp/rcpp-rcpparmadillo-and-os-x-mavericks-lgfortran-and-lquadmath-error/).

- If you run into a non-zero exit status error while installing, try running `Sys.setenv(R_REMOTES_NO_ERRORS_FROM_WARNINGS=TRUE)` first as per [this post](https://stackoverflow.com/questions/64993205/failing-to-install-github-package-had-non-zero-exit-status-error).

The main functions are described below:
- `sim.sfm` simulates model parameters and data from a sparse factor model.
- `gibbs` performs inference on the sparse factor model using a collapsed Gibbs sampler (MCMC).
- `cavi` performs inference on the sparse factor model using coordinate ascent variational inference.
- `relabel_samples` deals with label-switching and sign-switching of a MCMC chain. If there are multiple chains, they need to be concatenated together first.
- `relabel_params` relabels a realisation of the model parameters (e.g. posterior mean or simulated data) to match a target set of model parameters.

For details of the model, see `docs/assets/report.pdf`. The functions `gibbs` and `cavi` run much faster if there are no `NA`s in the data. Examples showing how to use these functions can be found in `scripts`. Documentation is available through calling `help` in R upon building the package, and also available in `man`.