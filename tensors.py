import torch
import numpy as np

class MyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input



N, D_in, H, D_out = 64, 1000, 100, 10

# x = np.random.randn(N, D_in)
# y = np.random.randn(N, D_out)
#
# w1 = np.random.randn(D_in, H)
# w2 = np.random.randn(H, D_out)
#
# learning_rate = 1e-6
# for t in range(500):
#     h = x.dot(w1)
#     h_relu = np.maximum(h, 0)
#     y_pred = h_relu.dot(w2)
#
#     loss = np.square(y_pred - y).sum()
#     print(t, loss)
#
#     # Backprop to compute gradients of w1 and w2 with respect to loss
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.T.dot(grad_y_pred)
#     grad_h_relu = grad_y_pred.dot(w2.T)
#     grad_h = grad_h_relu.copy()
#     grad_h[h < 0] = 0
#     grad_w1 = x.T.dot(grad_h)
#
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2

dtype = torch.float
device = torch.device("cpu")

# x = torch.randn(N, D_in, device=device, dtype=dtype)
# y = torch.randn(N, D_out, device=device, dtype=dtype)
#
# w1 = torch.randn(D_in, H, device=device, dtype=dtype)
# w2 = torch.randn(H, D_out, device=device, dtype=dtype)
#
# learning_rate = 1e-6
#
# for t in range(500):
#     h = x.mm(w1)
#     h_relu = h.clamp(min=0)
#     y_pred = h_relu.mm(w2)
#
#     loss = (y_pred - y).pow(2).sum().item()
#     print (t, loss)
#
#     # Backprop to compute gradients of w1 and w2 with respect to loss
#     grad_y_pred = 2.0 * (y_pred - y)
#     grad_w2 = h_relu.t().mm(grad_y_pred)
#     grad_h_relu = grad_y_pred.mm(w2.t())
#     grad_h = grad_h_relu.clone()
#     grad_h[h < 0] = 0
#     grad_w1 = x.t().mm(grad_h)
#     w1 -= learning_rate * grad_w1
#     w2 -= learning_rate * grad_w2


x = torch.randn(N, D_in, device=device, dtype=dtype, requires_grad=False)
y = torch.randn(N, D_out, device=device, dtype=dtype, requires_grad=False)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

# for t in range(500):
#     y_pred = x.mm(w1).clamp(min=0).mm(w2)
#
#     loss = (y_pred - y).pow(2).sum()
#     print(t, loss.item())
#
#     loss.backward() # computes gradients of loss wrt all the requires_grad=true automatically
#
#     # go through each weight for grad desc
#     with torch.no_grad():
#         # alternatively way is to operate on w1.data and w2.data
#         # w1.grad.data and w2.grad.data
#         # tensor.data - gives a tensor that shares the storage with tensor w/o history
#         w1 -= learning_rate * w1.grad
#         w2 -= learning_rate * w2.grad
#
#         # zero the tensor gradients - this is required for updating weights
#         w1.grad.zero_()
#         w2.grad.zero_()

for t in range(500):
    relu = MyRelu.apply # Function.apply method - aliased as "relu"

    y_pred = relu(x.mm(w1)).mm(w2)

    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    loss.backward()

    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()
        w2.grad.zero_()
