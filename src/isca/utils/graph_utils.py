import torch


def infer_soft_graph(assign: torch.Tensor):
    """
    assign : (b,n,k) soft cluster responsibilities
    returns soft adjacency (b,n,n) via cluster overlap.
    """
    return assign @ assign.transpose(1, 2)  # cluster inner product
