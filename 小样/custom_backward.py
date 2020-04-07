import torch
import numpy as np
from PIL import Image
from torch.autograd import gradcheck
class Bicubic(torch.autograd.Function):
    def basis_function(self, x, a=-1):
        x_abs = np.abs(x)
        if x_abs < 1 and x_abs >= 0:
            y = (a + 2) * np.power(x_abs, 3) - (a + 3) * np.power(x_abs, 2) + 1
        elif x_abs > 1 and x_abs < 2:
            y = a * np.power(x_abs, 3) - 5 * a * np.power(x_abs, 2) + 8 * a * x_abs - 4 * a
        else:
            y = 0
        return y
    def bicubic_interpolate(self,data_in, scale=1 / 4, mode='edge'):
        # data_in = data_in.detach().numpy()
        self.grad = np.zeros(data_in.shape,dtype=np.float32)
        obj_shape = (int(data_in.shape[0] * scale), int(data_in.shape[1] * scale), data_in.shape[2])
        data_tmp = data_in.copy()
        data_obj = np.zeros(shape=obj_shape, dtype=np.float32)
        data_in = np.pad(data_in, pad_width=((2, 2), (2, 2), (0, 0)), mode=mode)
        print(data_tmp.shape)
        for axis0 in range(obj_shape[0]):
            f_0 = float(axis0) / scale - np.floor(axis0 / scale)
            int_0 = int(axis0 / scale) + 2
            axis0_weight = np.array(
                [[self.basis_function(1 + f_0), self.basis_function(f_0), self.basis_function(1 - f_0), self.basis_function(2 - f_0)]])
            for axis1 in range(obj_shape[1]):
                f_1 = float(axis1) / scale - np.floor(axis1 / scale)
                int_1 = int(axis1 / scale) + 2
                axis1_weight = np.array(
                    [[self.basis_function(1 + f_1), self.basis_function(f_1), self.basis_function(1 - f_1), self.basis_function(2 - f_1)]])
                nbr_pixel = np.zeros(shape=(obj_shape[2], 4, 4), dtype=np.float32)
                grad_point = np.matmul(np.transpose(axis0_weight, (1, 0)), axis1_weight)
                for i in range(4):
                    for j in range(4):
                        nbr_pixel[:, i, j] = data_in[int_0 + i - 1, int_1 + j - 1, :]
                        for ii in range(data_in.shape[2]):
                            self.grad[int_0 - 2 + i - 1, int_1 - 2 + j - 1, ii] = grad_point[i,j]
                tmp = np.matmul(axis0_weight, nbr_pixel)
                data_obj[axis0, axis1, :] = np.matmul(tmp, np.transpose(axis1_weight, (1, 0)))[:, 0, 0]
                # img = np.transpose(img[0, :, :, :], [1, 2, 0])
        return data_obj

    def forward(self,input):
        print(type(input))
        input_ = input.detach().numpy()
        output = self.bicubic_interpolate(input_)
        # return input.new(output)
        return torch.Tensor(output)

    def backward(self,grad_output):
       print(self.grad.shape,grad_output.shape)
       grad_output.detach().numpy()
       grad_output_tmp = np.zeros(self.grad.shape,dtype=np.float32)
       for i in range(self.grad.shape[0]):
           for j in range(self.grad.shape[1]):
               grad_output_tmp[i,j,:] = grad_output[int(i/4),int(j/4),:]
       grad_input = grad_output_tmp*self.grad
       print(type(grad_input))
       # return grad_output.new(grad_input)
       return torch.Tensor(grad_input)

def bicubic(input):
    return Bicubic()(input)

def main():
	hr = Image.open('./baboon/baboon_hr.png').convert('L')
	hr = torch.Tensor(np.expand_dims(np.array(hr), axis=2))
	hr.requires_grad = True
	lr = bicubic(hr)
	print(lr.is_leaf)
	loss=torch.mean(lr)
	loss.backward()
if __name__ =='__main__':
	main()
