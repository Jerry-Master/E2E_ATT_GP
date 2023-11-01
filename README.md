# An End-to-end Approach to combine Attention feature extraction and Gaussian Process models for Deep Multiple Instance Learning in CT Hemorrhage Detection

Official release for the article published at Expert Systems With Applications. We provide the code used to obtain the published results.

## Installation

To run this code you will need to install the dependencies in `requirements.txt` in a python environment. Code has been tested on CentOS 7 and python7 but should work on any python version and OS that supports tensorflow.

## How to use

To run this code you will need to download the preprocessed RSNA dataset from the following links:

[https://www.kaggle.com/datasets/josepc/brain-ct](https://www.kaggle.com/datasets/josepc/brain-ct)
[https://www.kaggle.com/josepc/brain-ct-npy-1](https://www.kaggle.com/josepc/brain-ct-npy-1)
[https://www.kaggle.com/josepc/brain-ct-npy-2](https://www.kaggle.com/josepc/brain-ct-npy-2)
[https://www.kaggle.com/josepc/brain-ct-npy-3](https://www.kaggle.com/josepc/brain-ct-npy-3)
[https://www.kaggle.com/josepc/brain-ct-npy-test](https://www.kaggle.com/josepc/brain-ct-npy-test)

And modify lines 43 to 53 of `E2E_ATT_GP.py` and `E2E_GP_ATT.py` to point to the location of the files. You should modify line 204 and 207 respectively to indicate the location of where to save the resulting checkpoints. All the pretrained checkpoints are provided in the release. There is one model for each combination of hyperparameter employed.

## Implementation details

Most of the code is boilerplate for the training loop, the key part is the definition of the model:

```python
# E2E_ATT_GP
cnn_part = CNN_part()
att_part = attention_ATT_GP((dim[0]//64-2, dim[1]//64-2, 32))
kernel = gpflow.kernels.SquaredExponential(variance=0.5, lengthscales=[1.5])
inducing_variable = create_inducing_points(train_dataset2)
gp_layer = GPLayerSeq(
    kernel, inducing_variable, num_data=num_data * dim[2], num_latent_gps=output_dim, 
    mean_function=gpflow.mean_functions.Identity(), scale_factor=scale_factor
)
model = keras.Sequential([cnn_part,
                            att_part,
                            gp_layer,
                            layers.Dense(1, activation='sigmoid')])

# E2E_GP_ATT
cnn_part = CNN_part()
att_part = attention_GP_ATT((8), fc=False)
kernel = gpflow.kernels.SquaredExponential(variance=0.5, lengthscales=[1.5])
inducing_variable = create_inducing_points(train_dataset2)
gp_layer = GPLayerSeq(
    kernel, inducing_variable, num_data=num_data * dim[2], num_latent_gps=output_dim, 
    mean_function=gpflow.mean_functions.Identity(), scale_factor=scale_factor
)
model = keras.Sequential([cnn_part,
                        layers.Flatten(),
                        layers.Dense(8),
                        gp_layer,
                        att_part,
                        layers.Dense(1, activation='sigmoid')
                            ])
```

The `GPLayerSeq` is a custom Gaussian Process layer we needed to create to change the way the Kullback-Leibler divergence is added in the standard libraries. By default it was being scaled by the number of datapoints. We changed that to scale the KL loss by any amount we desired and called that amount the scaling factor. Other relevant aspect to bear in mind is the way the inducing points are initialized (`create_inducing_points`). We used a grid ranging between the minimum and the maximum value of the coordinates in the latent space. Also, the attention layer operates at a different latent dimension depending on whether it is used previous to the GP layer of after it. For a reference on the values of all the hyperparameters, please refer to the article.

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Citation