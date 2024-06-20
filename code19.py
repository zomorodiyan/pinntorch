

loss_bc = sum([weights['bc'][i] * bc_losses[i] for i in range(len(bc_losses))])
loss_sparse = sum([weights['sparse'][i] * sparse_losses[i] for i in range(len(sparse_losses))])

