import torch
from BDD.bdd_cuda_learned_mma_py import bdd_cuda_learned_mma


class DualIterations(torch.autograd.Function):
    @staticmethod
    def forward(ctx, solvers, lo_costs_batch, hi_costs_batch, delta_lo_batch, delta_hi_batch, dist_weights_batch, num_iterations, omega):
        assert(lo_costs_batch.is_contiguous())
        assert(hi_costs_batch.is_contiguous())
        assert(delta_lo_batch.is_contiguous())
        assert(delta_hi_batch.is_contiguous())
        assert(lo_costs_batch.dim() == 1)
        assert(delta_lo_batch.dim() == 1)
        assert(dist_weights_batch.dim() == 1)
        assert(dist_weights_batch.is_contiguous())
        assert(lo_costs_batch.shape == hi_costs_batch.shape)
        assert(lo_costs_batch.shape == dist_weights_batch.shape)
        assert(delta_lo_batch.shape == delta_hi_batch.shape)
        ctx.set_materialize_grads(False)
        layer_start = 0
        var_start = 0
        ctx.save_for_backward(lo_costs_batch, hi_costs_batch, delta_lo_batch, delta_hi_batch, dist_weights_batch, num_iterations, omega)
        ctx.solvers = solvers
        deff_mm_diff_out_batch = torch.empty_like(hi_costs_batch)
        lo_costs_out = torch.empty_like(lo_costs_batch)
        hi_costs_out = torch.empty_like(hi_costs_batch)
        delta_lo_out = torch.empty_like(delta_lo_batch)
        delta_hi_out = torch.empty_like(delta_hi_batch)

        for (b, solver) in enumerate(solvers):
            solver.set_solver_costs(lo_costs_batch[layer_start].data_ptr(),
                                    hi_costs_batch[layer_start].data_ptr(),
                                    delta_lo_batch[var_start].data_ptr(),
                                    delta_hi_batch[var_start].data_ptr())
            solver.iterations(dist_weights_batch[layer_start].data_ptr(),
                            deff_mm_diff_out_batch[layer_start].data_ptr(),
                            num_iterations,
                            omega)
            solver.get_solver_costs(lo_costs_out[layer_start].data_ptr(),
                                    hi_costs_out[layer_start].data_ptr(),
                                    delta_lo_out[var_start].data_ptr(),
                                    delta_hi_out[var_start].data_ptr())
            layer_start += solver.nr_layers()
            var_start += solver.nr_primal_variables()
        assert(layer_start == lo_costs_batch.shape[0])
        assert(var_start == delta_lo_batch.shape[0])
        # deltas are not backpropagated since deferred min-marginals contain the same information.
        # Only sent for computational efficiency to set the solver state later.
        ctx.mark_non_differentiable(delta_lo_out)
        ctx.mark_non_differentiable(delta_hi_out)
        return lo_costs_out, hi_costs_out, delta_lo_out, delta_hi_out, deff_mm_diff_out_batch

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_lo_costs_out, grad_hi_costs_out, grad_delta_lo_out, grad_delta_hi_out, grad_deff_mm_diff_out):
        assert(grad_lo_costs_out.is_contiguous())
        assert(grad_hi_costs_out.is_contiguous())
        assert(grad_delta_lo_out is None)
        assert(grad_delta_hi_out is None)
        assert(grad_deff_mm_diff_out.is_contiguous())
        assert(grad_lo_costs_out.dim() == 1)
        assert(grad_lo_costs_out.shape == grad_hi_costs_out.shape)
        assert(grad_lo_costs_out.shape == grad_deff_mm_diff_out.shape)

        grad_lo_costs_in = grad_lo_costs_out.detach().clone()
        grad_hi_costs_in = grad_hi_costs_out.detach().clone()
        grad_deff_mm_diff_in = grad_deff_mm_diff_out.detach().clone()
        solvers = ctx.solvers
        lo_costs_batch, hi_costs_batch, delta_lo_batch, delta_hi_batch, dist_weights_batch, num_iterations, omega = ctx.saved_tensors
        grad_dist_weights_batch_in = torch.empty_like(dist_weights_batch)

        layer_start = 0
        var_start = 0
        for (b, solver) in enumerate(solvers):
            solver.set_solver_costs(lo_costs_batch[layer_start].data_ptr(),
                                    hi_costs_batch[layer_start].data_ptr(),
                                    delta_lo_batch[var_start].data_ptr(),
                                    delta_hi_batch[var_start].data_ptr())

            solver.grad_iterations(dist_weights_batch[layer_start].data_ptr(),
                                    grad_lo_costs_in[layer_start].data_ptr(),
                                    grad_hi_costs_in[layer_start].data_ptr(),
                                    grad_deff_mm_diff_in[layer_start].data_ptr(),
                                    grad_dist_weights_batch_in[layer_start].data_ptr(),
                                    omega, 
                                    0, 
                                    num_iterations)
            layer_start += solver.nr_layers()
            var_start += solver.nr_primal_variables()
        return grad_lo_costs_in, grad_hi_costs_in, None, None, grad_dist_weights_batch_in, None, 