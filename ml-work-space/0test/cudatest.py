import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"사용 가능한 장치: {device}")
