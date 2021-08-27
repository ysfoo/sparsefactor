# sparsefactor: Bayesian inference techniques for sparse factor models

This project provides two approaches to inference for sparse factor models: Gibbs sampling and variational inference. Additionally, a relabelling algorithm is implemented to address issues which arise from label switching and sign switching.

The main functionality can be accessed by building an `R` package:

   ```R
   install.packages("devtools")
   library(devtools)
   install_github("ysfoo/sparsefactor")
   ```

See [here](https://ysfoo.github.io/sparsefactor) for an explanation of the project.
