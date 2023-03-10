import torch
import torch.nn as nn
from .InterpreterBase import Interpreter

class GuidedBackprop(Interpreter):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model):
        super(GuidedBackprop, self).__init__(model)
        self.forward_relu_outputs = []
        self.update_relus()

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            if len(self.forward_relu_outputs) != 0:
                corresponding_forward_output = self.forward_relu_outputs[-1]
                corresponding_forward_output[corresponding_forward_output > 0] = 1
                modified_grad_in = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
                del self.forward_relu_outputs[-1]  # Remove last forward output
                return (modified_grad_in,)
            else:
                print("Warning 0")


        def relu_forward_hook_function(module, input, output):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(output)


        # Loop through layers, hook up ReLUs
        for layer in self.model.named_modules():
            if 'classifier' not in layer[0] and isinstance(layer[1], nn.ReLU):
                self.handles.append(layer[1].register_backward_hook(relu_backward_hook_function))
                self.handles.append(layer[1].register_forward_hook(relu_forward_hook_function))

    def release(self):
        super(GuidedBackprop,self).release()
        self.forward_relu_outputs = []