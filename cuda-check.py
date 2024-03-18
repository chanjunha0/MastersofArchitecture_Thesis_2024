import torch

def check_cuda():
    if torch.cuda.is_available():
        print("CUDA is available on your system.")
        print("CUDA Device Count:", torch.cuda.device_count())
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    else:
        print("CUDA is not available on your system.")

if __name__ == "__main__":
    check_cuda()
