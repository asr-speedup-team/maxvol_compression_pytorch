# maxvol_compression_pytorch
Compression with maxvol

The method is based on the search of an optimal solution $$V$$ of

$$x_1^{T} = V_1 c_1^{T} = f(W_1 x_0^{T})$$          [1]

Here $$x_1$$ is the activation at layer 1, $$x_0 \in \mathbb{R}^{D_0 \times N}$$ is previous activation, $$W_1 \in \mathbb{R}^{D_{1} \times {D_0}}$$ is the weight matrix and $$c_1 \in \mathbb{R}^{R_1 \times N}$$ is a "low rank" component of $$x_1$$.

$$V_1$$ can be obtained from the singular value decomposition of $$x_1$$ across the whole dataset, e.g. if 

$$x_1 \in \mathbb{R}^{N \times D}$$  then

 $$x_1^{T} = V_1 \cdot c_1^{T}$$

where $$V_1 \in \mathbb{R}^{D_1 \times R_1},  c_1 \in \mathbb{R}^{R_1 \times N}$$ , and $$R_1 \leq D_1$$

From the latter, 

$$c_1^T = V_1^{\dagger} x_1^{T}$$                  [2]

However, we would like to obtain some approximate solution for $$c^{T}$$, which will lead to reduction of computational cost. For this matter we approximate 

 $$c_1^{T} \approx (S_1 V_1)^{\dagger} (S_1 x_1^{T})$$                 [3]

which will still yield good solution of an overdetermined system [2] in some metric (least squares or something else). Here $$S_1 \in \mathbb{R}^{Q_1 \times D_1}$$ is a selection matrix and can commute with pointwise activation $$f()$$.

$$S_1$$ can be found by the Maximal Volume algorithm. 

Putting everything back into [1] we obtain a reduced model:

$$x_1^{T} = V_1 (V_1^{\dagger} S_1^{\dagger}) f(S_1 W_1 x_0^{T}) = V_1 (V_1^{\dagger} S_1^{\dagger}) f(\tilde{W}_1 x_0^T)$$         [4] 

Here $$\tilde{W}_1 \in \mathbb{R}^{Q_1 \times D_0},  ~~~~ Q_{1} \leq D_{1}$$               

The procedure can be repeated recursively. For the $$k$$-th layer the new approximate weights are:

$$\tilde{W}_k = S_{k} W_{k} V_{k-1}  (V_{k-1}^{\dagger} S_{k-1}^{T})$$                    [5]

The new reduced weight has lower dimensions:

$$\tilde{W}_k \in \mathbb{R}^{Q_{k} \times Q_{k-1}},  ~~ Q_{k} \leq D_{k}, ~~ Q_{k-1} \leq D_{k-1}$$

This technique can be applied to dense layers, convolutional layers (by unrolling them into dense matrices) and recurrent layers (as they contain dense layers internally). 

After compression, the network can be fine-tuned to reduce error.

See notebooks/ExampleMaxvol.ipynb for a complete steo-to-step process.