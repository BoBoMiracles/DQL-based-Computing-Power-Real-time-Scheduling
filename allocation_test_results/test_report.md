# Compute Allocation Strategy Test Report

## Test Overview
This test evaluates the impact of different compute allocation strategies on reinforcement learning model performance.

## Test Results Summary
| Strategy | Success Rate | Avg Latency (ms) | Cloud Usage Rate | Convergence Episodes |
|----------|--------------|------------------|------------------|----------------------|
| original | 0.456 | 71.23 | 0.010 | 69 |
| uniform | 0.168 | 257.80 | 0.005 | 10 |
| hotspot | 0.366 | 96.51 | 0.000 | 16 |
| weighted | 0.038 | 1418.14 | 0.938 | 10 |
| random_seed42 | 0.493 | 62.26 | 0.000 | 92 |
| random_seed123 | 0.339 | 106.56 | 0.001 | 10 |
| random_seed456 | 0.368 | 95.88 | 0.004 | 61 |

## Best Strategy
Based on test results, **random_seed42** strategy performed best.

## Detailed Analysis
### original
Original optimal allocation

Allocation details:
```python
{'ACN_0': 11, 'ACN_1': 10, 'DCN_0': 3, 'DCN_1': 3, 'DCN_10': 3, 'DCN_11': 2, 'DCN_12': 3, 'DCN_13': 3, 'DCN_14': 2, 'DCN_15': 1, 'DCN_16': 1, 'DCN_17': 3, 'DCN_2': 3, 'DCN_3': 4, 'DCN_4': 3, 'DCN_5': 1, 'DCN_6': 4, 'DCN_7': 2, 'DCN_8': 3, 'DCN_9': 1, 'EC_0': 18}
```

### uniform
Uniform allocation (approx 4 boards per room)

Allocation details:
```python
{'ACN_0': 4, 'ACN_1': 4, 'DCN_0': 4, 'DCN_1': 4, 'DCN_10': 4, 'DCN_11': 4, 'DCN_12': 4, 'DCN_13': 4, 'DCN_14': 4, 'DCN_15': 4, 'DCN_16': 4, 'DCN_17': 4, 'DCN_2': 4, 'DCN_3': 4, 'DCN_4': 4, 'DCN_5': 4, 'DCN_6': 4, 'DCN_7': 4, 'DCN_8': 4, 'DCN_9': 4, 'EC_0': 4}
```

### hotspot
Hotspot concentration allocation

Allocation details:
```python
{'ACN_0': 15, 'ACN_1': 3, 'DCN_0': 4, 'DCN_1': 4, 'DCN_10': 3, 'DCN_11': 1, 'DCN_12': 2, 'DCN_13': 4, 'DCN_14': 5, 'DCN_15': 3, 'DCN_16': 6, 'DCN_17': 2, 'DCN_2': 4, 'DCN_3': 3, 'DCN_4': 3, 'DCN_5': 3, 'DCN_6': 6, 'DCN_7': 6, 'DCN_8': 1, 'DCN_9': 5, 'EC_0': 1}
```

### weighted
Request density weighted allocation

Allocation details:
```python
{'ACN_0': 53, 'ACN_1': 12, 'DCN_0': 1, 'DCN_1': 1, 'DCN_10': 1, 'DCN_11': 1, 'DCN_12': 1, 'DCN_13': 1, 'DCN_14': 1, 'DCN_15': 1, 'DCN_16': 1, 'DCN_17': 1, 'DCN_2': 1, 'DCN_3': 1, 'DCN_4': 1, 'DCN_5': 1, 'DCN_6': 1, 'DCN_7': 1, 'DCN_8': 1, 'DCN_9': 1, 'EC_0': 1}
```

### random_seed42
Random allocation (seed=42)

Allocation details:
```python
{'ACN_0': 3, 'ACN_1': 5, 'DCN_0': 5, 'DCN_1': 4, 'DCN_10': 2, 'DCN_11': 2, 'DCN_12': 2, 'DCN_13': 5, 'DCN_14': 4, 'DCN_15': 4, 'DCN_16': 2, 'DCN_17': 17, 'DCN_2': 5, 'DCN_3': 3, 'DCN_4': 2, 'DCN_5': 2, 'DCN_6': 3, 'DCN_7': 4, 'DCN_8': 3, 'DCN_9': 3, 'EC_0': 4}
```

### random_seed123
Random allocation (seed=123)

Allocation details:
```python
{'ACN_0': 4, 'ACN_1': 3, 'DCN_0': 3, 'DCN_1': 4, 'DCN_10': 4, 'DCN_11': 3, 'DCN_12': 18, 'DCN_13': 4, 'DCN_14': 3, 'DCN_15': 3, 'DCN_16': 3, 'DCN_17': 4, 'DCN_2': 3, 'DCN_3': 2, 'DCN_4': 3, 'DCN_5': 4, 'DCN_6': 2, 'DCN_7': 2, 'DCN_8': 4, 'DCN_9': 4, 'EC_0': 4}
```

### random_seed456
Random allocation (seed=456)

Allocation details:
```python
{'ACN_0': 3, 'ACN_1': 2, 'DCN_0': 4, 'DCN_1': 5, 'DCN_10': 4, 'DCN_11': 4, 'DCN_12': 17, 'DCN_13': 4, 'DCN_14': 2, 'DCN_15': 2, 'DCN_16': 3, 'DCN_17': 3, 'DCN_2': 4, 'DCN_3': 2, 'DCN_4': 4, 'DCN_5': 3, 'DCN_6': 4, 'DCN_7': 4, 'DCN_8': 4, 'DCN_9': 4, 'EC_0': 2}
```

## Conclusions and Recommendations
1. Compute allocation has a significant impact on model performance
2. It is recommended to optimize compute allocation based on actual request distribution
3. Regularly re-evaluate and adjust allocation strategies
