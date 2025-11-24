
data {
  int<lower=1> N;
  int<lower=1> I;
  int<lower=1> K;
  int<lower=1> C;
  int<lower=1> J;
  array[N] int<lower=1, upper=C> y;
  array[N] int<lower=1, upper=J> context;
  array[N] int<lower=1, upper=I> person;
  array[C, K, J] real<lower=0, upper=1> emission_probs;
  vector<lower=0>[K] alpha;
}
parameters {
  simplex[K] pi;
}
model {
  pi ~ dirichlet(alpha);

  for (i in 1:I) {
    vector[K] log_lik;
    for (k in 1:K) {
      real acc = 0;
      for (n in 1:N) {
        if (person[n] == i) {
          vector[C] p = to_vector(emission_probs[:, k, context[n]]);
          acc += categorical_lpmf(y[n] | p / sum(p));
        }
      }
      log_lik[k] = log(pi[k]) + acc;
    }
    target += log_sum_exp(log_lik);
  }
}
