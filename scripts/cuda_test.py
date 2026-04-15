import torch

def test_cuda():
    x = torch.rand(5, 3)
    print(x)
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device = torch.device("cuda")
        y = torch.ones_like(x, device=device)
        x = x.to(device)
        z = x + y
        print(z)
        print(z.to("cpu", torch.double))

if __name__ == "__main__":
    test_cuda()