from tritonclient.grpc import InferenceServerClient, InferInput, InferRequestedOutput
import numpy as np

# Initialize Triton gRPC client
client = InferenceServerClient(url='localhost:8001', verbose=True)

# Create input data (1x16 array filled with ones)
input_data = np.ones((1, 16), dtype=np.float32)

# Create InferInput object
input = InferInput('INPUT', input_data.shape, 'FP32')

# Set data for inference
input.set_data_from_numpy(input_data)

# Create InferRequestedOutput object
output = InferRequestedOutput('OUTPUT')

# Run inference
results = client.infer(model_name='multipy_method', inputs=[input], outputs=[output])

# Get results
output_data = results.as_numpy('OUTPUT')

# Print results
print("Input data: ", input_data)
print("Output data: ", output_data)

