# NALU
An implementation of Neural Arithmetic Logic Units (https://arxiv.org/pdf/1808.00508.pdf)

## Exp 1: Fail to Extrapolate
Train an identity mapping on [-5, 5] and test it on [-20, 20]

```bash
python3 failure.py
```

#### Results

* Most non linear activation functions fail to exprapolate except that PReLU can learn to be highly linear.

![Failure](failure.png)

## Exp 2: Static Simple Function Learning
Input a 100-dimensional vertex **x**, learn `y = func(a, b)`,
where <img src="https://latex.codecogs.com/svg.latex?a=\sum_{i=N}^{M}(\mathbf{x}_i)" title=""/>
, <img src="https://latex.codecogs.com/svg.latex?b=\sum_{i=P}^{Q}(\mathbf{x}_i)" title=""/>  and `func = +, -, x, /, ...`. Test the ability to interpolate and extrapolate.

```bash
python3 learn_function.py
```
### Interpolation
* RMSE (normalized to a random baseline)

|     |ReLU|Sigmoid|NAC|NALU|
| --- |  --- | --- | --- | --- |
|a + b|0.00|0.11|0.00|0.01|
|a - b|0.16|0.85|0.00|0.12|
|a x b|1.86|1.21|13.42|0.00|
|a / b|0.88|0.12|1.89|0.01|
|a ^ 2|3.56|0.32|20.56|0.00|
|sqrt(a)|0.60|0.14|2.56|0.02|

### Extrapolation
* RMSE (normalized to a random baseline)

|     |ReLU|Sigmoid|NAC|NALU|
| --- |  --- | --- | --- | --- |
|a + b|0.00|62.55|0.00|0.42|
|a - b|59.23|60.64|0.00|0.43|
|a x b|57.13|88.27|75.73|0.00|
|a / b|3.07|1.32|23.82|0.36|
|a ^ 2|57.99|81.51|76.48|0.00|
|sqrt(a)|16.58|18.08|63.17|0.17|


## Notes

### Differences between this implementation and the original one in the paper
* Use two separate weight matrices for the adder and the multiplier
* The gate is independent of the input

See [nalu.py](nalu.py) for more details. I found these modifications can help the performance on the static simple function learning task.

