class Net(torch.nn.Module):
    """We create neural nets by subclassing the torch.nn.Module class in
    the newly defined class, we define 2 things:
    1) The network elements/layers; these are defined in the __init__ method
    2) The network behavior! What happens when we call our model on an input
    (here we call the input 'x')

    Our model is thus composed of 2 Conv and 2 Linear layers, each with a
    different size. When we call our model against an input example, we compute
    the output from each layer and in between we apply the ReLU function.
    """

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5)
        self.layer2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5)
        self.layer3 = torch.nn.Linear(800, 16)
        self.layer4 = torch.nn.Linear(16, 10)

    def forward(self, x):
        # pass through conv layers
        x = self.layer1(x)
        x = torch.nn.functional.relu(x)
        x = self.layer2(x)
        x = torch.nn.functional.relu(x)

        # pass through linear layers
        x = torch.flatten(x, start_dim=1)  # flatten the output of convolution
        x = self.layer3(x)
        x = torch.nn.functional.relu(x)
        x = self.layer4(x)
        return x


# we initialize our model as thus:
# Congrats! You just built a neural network with PyTorch :-)
my_model = Net()


# GPU-aware programming
"""
our PyTorch module loads automatically to CPU, and afterwards we can decide to
send it to GPU using .to() method. In fact tensor.to() method can send any
PyTorch tensor to any device, not just models.
"""
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(DEVICE)