from typing import Any, Dict, List, Optional, Tuple
import gpflow
from gpflow import default_float
from gpflow.base import TensorType
from gpflow.inducing_variables import MultioutputInducingVariables
from gpflow.kernels import MultioutputKernel
from gpflow.mean_functions import MeanFunction
import gpflux
import tensorflow as tf

class GPLayerSeq(gpflux.layers.gp_layer.GPLayer):

    def __init__(
        self,
        kernel: MultioutputKernel,
        inducing_variable: MultioutputInducingVariables,
        num_data: int,
        mean_function: Optional[MeanFunction] = None,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
        num_latent_gps: int = None,
        whiten: bool = True,
        name: Optional[str] = None,
        verbose: bool = True,
        scale_factor: float = 1
    ):
        super().__init__(
            kernel=kernel,
            inducing_variable=inducing_variable,
            num_data=num_data,
            mean_function=mean_function,
            num_samples=num_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            num_latent_gps=num_latent_gps,
            whiten=whiten,
            name=name,
            verbose=verbose,
        )
        self.scale_factor = scale_factor 

    def call(self, inputs: TensorType, *args: List[Any], **kwargs: Dict[str, Any]) -> tf.Tensor:
        outputs = super(gpflux.layers.gp_layer.GPLayer, self).call(inputs, *args, **kwargs)

        if kwargs.get("training"):
            log_prior = tf.add_n([p.log_prior_density() for p in self.kernel.trainable_parameters])
            loss = self.prior_kl()
            loss_per_datapoint = self.scale_factor * loss

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_KL" if self.name else "prior_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return outputs
