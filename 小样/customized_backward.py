import torch.nn.functional as Function
import torch.nn as nn

class my_function(Function):
 def forward(self, input, parameters):
        self.saved_for_backward = [input, parameters]
        # output = [对输入和参数进行的操作，这里省略]
        return output
 def backward(self, grad_output):
        input, parameters = self.saved_for_backward
        # grad_input = [求 forward(input)关于 parameters 的导数] * grad_output
        return grad_input
# 然后通过定义一个Module来包装一下
class my_module(nn.Module):
      def __init__(self, ...):
         super(my_module, self).__init__()
         self.parameters = # 初始化一些参数
      def backward(self, input):
          output = my_function(input, self.parameters) # 在这里执行你之前定义的function!
          return output