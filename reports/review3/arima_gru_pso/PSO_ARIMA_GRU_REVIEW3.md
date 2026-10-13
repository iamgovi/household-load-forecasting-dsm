# PSO-tuned ARIMA–GRU — Review 3

## Search and leakage controls

PSO used 5 particles for 5 iterations (25 logged evaluations; 13 unique configurations), fixed seed 42 and four screening epochs. Fitness was RMSE on a chronological 80/20 inner split of training windows. Main validation and test data did not influence particle fitness; test was not loaded. The Milestone 4 residual pipeline and ARIMA(3,0,0) were unchanged.

## Selected configuration

PSO selected hidden=64, learning rate=0.002, dropout=0.3, lookback=24 and batch size=64. Inner-validation RMSE was 0.3577. Search runtime was 172.30s.

## Main validation result

On the same 1,794 origins and 58 meters, tuned seed 42 obtained MAE 0.1771 and RMSE 0.2807, versus untuned MAE 0.1798 and RMSE 0.2869. This is an absolute RMSE reduction of 0.0062 (2.15%) and MAE reduction of 0.0026 (1.45%).

## Seed, paired and household evidence

Tuned RMSE across three seeds was 0.2810 ± 0.0009. Untuned RMSE across two seeds was 0.2873 ± 0.0006. The tuned-minus-untuned paired-origin RMSE difference averaged -0.0052, with 95% CI [-0.0062, -0.0042] and tuned win rate 63.4%. PSO improved 41/58 meters, worsened 17/58 and tied 0/58.

## Conclusion

**PSO classification: MARGINALLY USEFUL.** The gain exceeds observed seed variability and paired evidence supports it, but the search adds optimization cost and the tuned hybrid still does not surpass Linear Regression or XGBoost. Final architecture selection and test evaluation must wait. The defensible final-selection candidates are Linear Regression, XGBoost, untuned ARIMA-GRU and PSO-ARIMA-GRU; attention remains a negative ablation.

## Limitations

The discrete PSO space and 5×5 budget are intentionally small. Screening uses only four epochs, so fitness is a resource-bounded proxy. Untuned variability has two seeds rather than three due CPU cost. Main validation remains selection-stage evidence, and `t_kWh` semantics remain authoritatively unverified.
