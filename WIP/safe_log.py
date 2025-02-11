from braindecode.models.functions import (
    safe_log,
)
from torch import Tensor, nn, from_numpy

class SafeLog(nn.Module):
    r"""
    Safe logarithm activation function module.

    :math:\text{SafeLog}(x) = \log\left(\max(x, \epsilon)\right)

    Parameters
    ----------
    eps : float, optional
        A small value to clamp the input tensor to prevent computing log(0) or log of negative numbers.
        Default is 1e-6.

    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x) -> Tensor:
        """
        Forward pass of the SafeLog module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying safe logarithm.
        """
        return safe_log(x=x, eps=self.eps)

    def extra_repr(self) -> str:
        eps_str = f"eps={self.eps}"
        return eps_str