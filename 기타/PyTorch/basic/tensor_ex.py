import torch
import numpy as np

# 데이터로부터 직접 생성
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# Numpy 배열로부터 생성
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 텐서로부터 생성
x_ones = torch.ones_like(x_data)    # x_data의 속성을 유지
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다
print(f"Random Tensor: \n {x_ones} \n")

# 무작위(random) 또는 상수(constant) 값 사용
# shape은 텐서의 차원을 나타내는 튜플로, 출력 텐서의 차원을 결정한다
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 텐서의 속성
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 텐서 연산
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")
tensor[:, 1] = 0
print(tensor)

# 텐서 합치기
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)





