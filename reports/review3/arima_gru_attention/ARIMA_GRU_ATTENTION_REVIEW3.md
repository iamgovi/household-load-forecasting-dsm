# ARIMA–GRU temporal-attention ablation — Review 3

## Architecture and preserved residual pipeline

The model adds additive temporal softmax attention across all 168 hidden states of the frozen
one-layer, 32-hidden-unit residual GRU. All other principal training settings are unchanged. The
286,079 residuals regenerated through the Milestone 4 function matched the frozen residual file,
and protected baseline hashes were unchanged. No test data, PSO, spatial attention or future fill
was used.

## Validation result

All 1,794 hybrid-compatible origins across 58 meters were evaluated. Attention obtained MAE
0.2012, RMSE 0.3110, sMAPE 62.12%, R² 0.4650 and NRMSE
0.7758. The plain hybrid retained MAE 0.1798 and RMSE 0.2869.
Attention changed MAE by -0.0214 (-11.91%)
and RMSE by -0.0241 (-8.40%); negative
improvement means degradation.

## Paired and household evidence

Origin-level attention-minus-hybrid MAE averaged 0.0214
(95% bootstrap CI [0.0198, 0.0232]); attention
won 18.2% of origins. For RMSE the mean difference was
0.0197 and win rate 20.8%.
Attention improved 3/58 meters, worsened 55/58,
and was approximately tied on 0/58.

## Attention weights and interpretation

Weights are descriptive model allocations, not causal effects. Region-level mass and all 168-lag
aggregate weights are saved. Mean normalized entropy was 0.996 (1.0 is uniform), and the most
recent 24 hours received 14.35% of mass versus 14.29% under a uniform distribution. Therefore no
strong concentration pattern was observed. Visual structure alone cannot justify attention.

## Complexity and decision

The attention model has 5,240 parameters versus 4,152 for the plain
hybrid. Total residual-generation plus training time was 111.38s versus
230.08s; inference was 0.146 versus 0.091ms/origin.

**Decision: ATTENTION NOT JUSTIFIED.** Attention degraded the main validation metrics and added complexity.
It should remain documented as a negative ablation, not be assumed as the preferred architecture.
PSO should not tune this attention model merely because it appears in the provisional title. If PSO
is pursued next, the defensible target is the plain residual hybrid (or another validation-supported
candidate), with attention retained as a documented negative ablation.

## Limitations

This is validation-stage evidence with validation-based early stopping. Five original canonical
origins were already excluded by the common residual-history constraint. `t_kWh` semantics remain
authoritatively unverified. Attention weights do not establish causality.
