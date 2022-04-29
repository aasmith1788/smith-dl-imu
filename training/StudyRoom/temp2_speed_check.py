import torch
import torchvision
from sklearn.metrics import classification_report
from tqdm import tqdm


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
my_model.to(DEVICE)  # this sends the model to the appropriate device

# Get data
DATA_PATH = "data/FashionMNIST/"
# transforms
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
    ]
)
train_ds = torchvision.datasets.FashionMNIST(
    root=DATA_PATH, train=True, transform=transform, download=True
)
test_ds = torchvision.datasets.FashionMNIST(
    root=DATA_PATH, train=False, transform=transform, download=True
)

"""Let's define our training data loader. We will adopt a batch size of 16.
Shuffling the data is also useful for training. 0 workers means that the data
will be loaded in the main process.
"""
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=128, shuffle=True, num_workers=0
)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=128, shuffle=False, num_workers=0
)

# Let's create a translation from the class numbers to Human-Readable
# (and meaningful) text labels

LABEL_DICT = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}

"""There are many Optimizers in the PyTorch Library. We select the ADAM
optimizer as one of the most recognizable and efficient optimizers in the
Deep Learning field. We feed the optimizer object, the parameters which it
will optimize!
"""
optimizer = torch.optim.Adam(my_model.parameters())

"""There are many Loss functions in the PyTorch Library. We pick one that is
suitable for the problem we are working on (Image Classification). It is called
Cross Entropy Loss.
"""
criterion = torch.nn.CrossEntropyLoss()

"""We leverage the classification_report function in Sci-Kit Learn! More info
here: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
SK Learn has many other metrics, you are welcome to check them out and use as
needed for your problem.
"""


def compute_metrics(labels, preds, mode="train"):
    """print out standard metrics for classification."""
    # we use the values in the dictionary we defined earlier (see above)
    # to define a list of class names
    # this is helpful in making the report more meaningful!
    class_names = list(LABEL_DICT.values())
    labels = torch.cat(labels)  # concatenate list into tensor
    preds = torch.cat(preds)
    metrics = classification_report(
        y_true=labels,
        y_pred=preds,
        target_names=class_names,
        output_dict=True,
    )
    print(f"\n-------- begin report: {mode} ----------------------------\n")
    print(f"prediction sample: {preds[0:10]}, label sample: {labels[0:10]}")
    for key, value in metrics.items():
        print(key, value)
    print(f"\n-------- end report: {mode} -------------------------------\n")
    return metrics


def train_step(my_model, images, labels, optimizer, criterion):
    # zero out the gradient parameters
    optimizer.zero_grad()

    # forward + backward + optimize

    # forward step: compute model output
    prediction = my_model(images)

    # backward step: compute loss
    loss = criterion(prediction, labels)
    loss.backward()  # calculate mini-batch grads

    # optimizer step: update the model parameters
    optimizer.step()  # update weights afters mini-batch
    return prediction, loss


NUM_EPOCHS = 5  # how many times will we go over our data?

"""Some Notes:
1) tqdm is a helpful tool that provides nice "progress bar" graphic in the
output console :-) our work now looks professional
2) Models have 2 modes, training and evaluation. Since we're going to train
our model, we change the mode to train
"""
my_model.train()  # change model from eval mode to train mode

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0  # reset epoch loss
    all_preds = []  # list of all predictions in the epoch
    all_labels = []  # list of all labels in the epoch
    num_batches = len(train_loader)
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        # get data
        images, labels = batch
        images = images.to(device=DEVICE, dtype=torch.float)
        labels = labels.to(device=DEVICE, dtype=torch.long)
        # we defined "device" when we defined the model, see above

        prediction, loss = train_step(my_model, images, labels, optimizer, criterion)
        epoch_loss += loss  # compile batch loss

        # metrics
        """Get the classification using the argmax function and send the tensor
        to CPU using .cpu() (or do nothing if they're already in CPU). This is
        important, our compute_metrics function expects CPU tensors
        """
        preds = torch.argmax(prediction, dim=1).cpu()
        labels = labels.cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        # batch ends

    # epoch ends
    epoch_loss /= num_batches
    print("==================== train ==============")
    print(f"\n Train Loss: {epoch_loss} \n")

    # Compute metrics for classification
    metrics = compute_metrics(all_labels, all_preds)
print("finished training!")
# Congrats! you just trained your first neural network in PyTorch


# evaluation
my_model.eval()
with torch.no_grad():
    val_loss = 0.0  # reset epoch loss
    all_preds = []  # list of all predictions in the epoch
    all_labels = []  # list of all labels in the epoch
    num_batches = len(test_loader)
    for batch in tqdm(test_loader, desc=f"validation progress"):
        # get data
        images, labels = batch
        images = images.to(device=DEVICE, dtype=torch.float)
        labels = labels.to(device=DEVICE, dtype=torch.long)
        # we defined "device" when we defined the model, see above

        prediction = my_model(images)
        loss = criterion(prediction, labels)
        val_loss += loss  # compile batch loss

        # metrics
        preds = torch.argmax(prediction, dim=1).cpu()
        labels = labels.cpu()
        all_preds.append(preds)
        all_labels.append(labels)
        # batch ends

print("==================== evaluation ==============")
print("\n Validation Loss: %f \n" % val_loss)

metrics = compute_metrics(all_labels, all_preds, mode="val")