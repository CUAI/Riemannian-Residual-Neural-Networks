import torch


EPS = {torch.float32: 1e-8, torch.float64: 1e-8}


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + EPS[x.dtype], 1 - EPS[x.dtype])
        ctx.save_for_backward(x)
        res = (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


class Arsinh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        positive_case = x + torch.sqrt(1 + x.pow(2))
        negative_case = 1 / (torch.sqrt(1 + x.pow(2)) - x)
        return torch.where(x > 0, positive_case, negative_case).log()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 + input ** 2) ** 0.5


class Acosh(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        x = torch.clamp(x, min=1+EPS[x.dtype])
        z = torch.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        return torch.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z.data.clamp(min=EPS[z.dtype])
        z = g / z
        return z, None


artanh = Artanh.apply
arsinh = Arsinh.apply
arcosh = Acosh.apply

def tanh(x):
    return x.tanh()
