import numpy as np
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        
        for request in requests:
            in_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            in_numpy = in_tensor.as_numpy()
            
            # Perform computation (multiply by 2)
            out_numpy = in_numpy * 2
            
            out_tensor = pb_utils.Tensor("OUTPUT", out_numpy)
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor])
            responses.append(response)
        
        return responses

