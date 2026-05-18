# Moderator Analysis of Frame Effects

Moderator analysis over paired frame deltas. P-values for categorical
moderators are label-permutation ANOVA tests on paired deltas; model-size
trends use six model-level mean deltas and should be read descriptively.

## Permutation ANOVA by Moderator

| frame           | metric        | moderator     |   levels |   f_stat |   perm_p |   eta2 |
|:----------------|:--------------|:--------------|---------:|---------:|---------:|-------:|
| brainstorming   | overall_delta | scenario_type |        3 |    2.842 |    0.058 |  0.039 |
| brainstorming   | overall_delta | model_name    |        6 |    1.098 |    0.360 |  0.038 |
| brainstorming   | overall_delta | model_family  |        3 |    1.411 |    0.253 |  0.020 |
| brainstorming   | overall_delta | category      |        8 |    0.955 |    0.462 |  0.047 |
| brainstorming   | harm_delta    | scenario_type |        3 |    0.470 |    0.698 |  0.007 |
| brainstorming   | harm_delta    | model_name    |        6 |    0.642 |    0.690 |  0.023 |
| brainstorming   | harm_delta    | model_family  |        3 |    1.292 |    0.323 |  0.018 |
| brainstorming   | harm_delta    | category      |        8 |    1.143 |    0.361 |  0.056 |
| critical_review | overall_delta | scenario_type |        3 |    0.196 |    0.823 |  0.003 |
| critical_review | overall_delta | model_name    |        6 |    3.020 |    0.014 |  0.099 |
| critical_review | overall_delta | model_family  |        3 |    2.641 |    0.078 |  0.036 |
| critical_review | overall_delta | category      |        8 |    1.189 |    0.323 |  0.058 |
| critical_review | harm_delta    | scenario_type |        3 |    0.496 |    0.688 |  0.007 |
| critical_review | harm_delta    | model_name    |        6 |    1.148 |    0.348 |  0.040 |
| critical_review | harm_delta    | model_family  |        3 |    0.212 |    0.871 |  0.003 |
| critical_review | harm_delta    | category      |        8 |    1.136 |    0.358 |  0.055 |

## Model Mean Deltas

| frame           | model_name                                   | model_short           | model_family   |   params_b |   overall_delta |   harm_delta |
|:----------------|:---------------------------------------------|:----------------------|:---------------|-----------:|----------------:|-------------:|
| brainstorming   | huggingface/Qwen/Qwen2.5-0.5B-Instruct       | Qwen2.5-0.5B-Instruct | Qwen           |      0.500 |          -0.146 |        0.292 |
| brainstorming   | huggingface/Qwen/Qwen2.5-7B-Instruct         | Qwen2.5-7B-Instruct   | Qwen           |      7.000 |          -0.354 |        0.292 |
| brainstorming   | huggingface/google/gemma-3-1b-it             | gemma-3-1b-it         | Gemma          |      1.000 |          -0.396 |        0.500 |
| brainstorming   | huggingface/google/gemma-4-E2B-it            | gemma-4-E2B-it        | Gemma          |      2.000 |          -0.375 |        0.375 |
| brainstorming   | huggingface/meta-llama/Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | Llama          |      8.000 |          -0.354 |        0.292 |
| brainstorming   | huggingface/meta-llama/Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct | Llama          |      3.000 |          -0.500 |        0.250 |
| critical_review | huggingface/Qwen/Qwen2.5-0.5B-Instruct       | Qwen2.5-0.5B-Instruct | Qwen           |      0.500 |          -0.021 |       -0.125 |
| critical_review | huggingface/Qwen/Qwen2.5-7B-Instruct         | Qwen2.5-7B-Instruct   | Qwen           |      7.000 |          -0.229 |        0.167 |
| critical_review | huggingface/google/gemma-3-1b-it             | gemma-3-1b-it         | Gemma          |      1.000 |          -0.083 |        0.167 |
| critical_review | huggingface/google/gemma-4-E2B-it            | gemma-4-E2B-it        | Gemma          |      2.000 |           0.312 |        0.000 |
| critical_review | huggingface/meta-llama/Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct | Llama          |      8.000 |           0.042 |        0.083 |
| critical_review | huggingface/meta-llama/Llama-3.2-3B-Instruct | Llama-3.2-3B-Instruct | Llama          |      3.000 |           0.000 |       -0.042 |

## Model Size Trends

| frame           | metric        |   pearson_r |   pearson_p |   spearman_r |   spearman_p |   n_models |
|:----------------|:--------------|------------:|------------:|-------------:|-------------:|-----------:|
| brainstorming   | overall_delta |      -0.508 |       0.304 |       -0.058 |        0.913 |          6 |
| brainstorming   | harm_delta    |      -0.406 |       0.424 |       -0.395 |        0.439 |          6 |
| critical_review | overall_delta |      -0.151 |       0.776 |        0.143 |        0.787 |          6 |
| critical_review | harm_delta    |       0.508 |       0.304 |        0.406 |        0.425 |          6 |

## Omnibus Delta ANOVA

OLS ANOVA over paired deltas with frame-by-moderator terms.

| term                      |   sum_sq |      df |       F |   PR(>F) | metric        |
|:--------------------------|---------:|--------:|--------:|---------:|:--------------|
| C(frame)                  |    9.209 |   1.000 |  33.403 |    0.000 | overall_delta |
| C(scenario_type)          |    0.450 |   2.000 |   0.815 |    0.444 | overall_delta |
| C(model_family)           |    0.082 |   2.000 |   0.149 |    0.862 | overall_delta |
| C(category)               |    3.277 |   7.000 |   1.698 |    0.110 | overall_delta |
| C(frame):C(scenario_type) |    1.283 |   2.000 |   2.327 |    0.100 | overall_delta |
| C(frame):C(model_family)  |    1.938 |   2.000 |   3.514 |    0.031 | overall_delta |
| C(frame):C(category)      |    0.923 |   7.000 |   0.478 |    0.850 | overall_delta |
| log_params_b              |    0.280 |   1.000 |   1.017 |    0.314 | overall_delta |
| C(frame):log_params_b     |    0.018 |   1.000 |   0.064 |    0.800 | overall_delta |
| Residual                  |   72.233 | 262.000 | nan     |  nan     | overall_delta |
| C(frame)                  |    6.125 |   1.000 |  20.407 |    0.000 | harm_delta    |
| C(scenario_type)          |    0.437 |   2.000 |   0.729 |    0.483 | harm_delta    |
| C(model_family)           |    1.121 |   2.000 |   1.868 |    0.157 | harm_delta    |
| C(category)               |    2.597 |   7.000 |   1.236 |    0.283 | harm_delta    |
| C(frame):C(scenario_type) |    0.146 |   2.000 |   0.243 |    0.784 | harm_delta    |
| C(frame):C(model_family)  |    0.065 |   2.000 |   0.109 |    0.897 | harm_delta    |
| C(frame):C(category)      |    2.153 |   7.000 |   1.025 |    0.414 | harm_delta    |
| log_params_b              |    0.381 |   1.000 |   1.271 |    0.261 | harm_delta    |
| C(frame):log_params_b     |    0.483 |   1.000 |   1.609 |    0.206 | harm_delta    |
| Residual                  |   78.636 | 262.000 | nan     |  nan     | harm_delta    |

## Short Summary

- `brainstorming`:
  - strongest categorical heterogeneity for `overall_delta` is `scenario_type` (permutation p=0.058, eta2=0.039).
  - strongest categorical heterogeneity for `harm_delta` is `model_family` (permutation p=0.323, eta2=0.018).
  - model-size trend for `overall_delta`: Spearman rho=-0.058, p=0.913 over six models.
  - model-size trend for `harm_delta`: Spearman rho=-0.395, p=0.439 over six models.
- `critical_review`:
  - strongest categorical heterogeneity for `overall_delta` is `model_name` (permutation p=0.014, eta2=0.099).
  - strongest categorical heterogeneity for `harm_delta` is `model_name` (permutation p=0.348, eta2=0.040).
  - model-size trend for `overall_delta`: Spearman rho=0.143, p=0.787 over six models.
  - model-size trend for `harm_delta`: Spearman rho=0.406, p=0.425 over six models.
