# Convolutional Neural Networks

## 0 From multilayer networks to CNNs: Key terminology

Before diving into CNNs, let's clarify how they relate to the multilayer networks you already know:

### Familiar concepts that remain the same
- **Neurons**: Still computational units that sum inputs and apply activation functions
- **Weights**: Still learnable parameters that get updated during training
- **Activation functions**: Still non-linear functions (ReLU, sigmoid, tanh) applied after weighted sums
- **Hidden layers**: Still intermediate layers between input and output
- **Backpropagation**: Still the algorithm for computing gradients and updating weights

### New CNN-specific terminology
- **Kernel/Filter**: A small matrix of weights (like a 3×3 grid) that slides across the input. Think of it as a specialized neuron that looks at small patches instead of all inputs at once.
- **Convolution**: The sliding operation where the kernel multiplies with input patches. It's like having many neurons that share the same weights but look at different parts of the input.
- **Feature map**: The output of applying a kernel across the input. It's like a hidden layer, but organized as a 2D grid instead of a 1D vector.
- **Pooling**: A downsampling operation that reduces the size of feature maps (like taking the maximum value in each 2×2 region).
- **Padding**: Adding zeros around the input border so the kernel can process edge pixels properly.
- **Stride**: How many pixels the kernel moves each step (stride=1 means move one pixel at a time).

### The key insight
Instead of connecting every neuron to every input (like in multilayer networks), CNNs use **local connectivity** (neurons only see small patches) and **weight sharing** (the same kernel/weights are used across all patches). This dramatically reduces parameters while respecting spatial structure.

## 1 Biological and historical roots  
Hubel and Wiesel showed that neurons in a cat's visual cortex are activated by small oriented patches of the retinal image; deeper cortical layers respond to ever more complex combinations of those primitive edges. LeNet-5 adopted that principle in its layered "shared-weight" architecture for digit recognition, and every modern CNN—AlexNet, VGG, ResNet, EfficientNet—still uses the same sparse local connectivity and weight sharing that emerged from those early insights.

## 2 Why CNNs vs. Fully Connected Networks?

**The fundamental problem**: A 224×224 RGB image has 150,528 pixels. A fully connected layer (where every neuron connects to every input, like in multilayer networks) would require 150,528 × hidden_units parameters just for the first layer—millions of parameters that learn arbitrary correlations without respecting spatial structure.

**CNNs solve this through**:
- **Local connectivity**: Each neuron connects only to a small spatial patch (like 3×3 pixels) instead of all inputs
- **Weight sharing**: The same kernel (set of weights) is applied across all spatial locations, like using the same "template" everywhere
- **Translation equivariance**: Shifting the input shifts the output predictably (if you move a cat in the image, the "cat detector" response moves correspondingly)

**Comparison**:
- **Fully connected layer**: 224×224×3 → 1000 hidden units = 150M parameters (every neuron connects to every input)
- **Convolutional layer**: 3×3×3 → 64 filters = 1,728 parameters (87,000× fewer!) (each filter is a small 3×3 template)

## 3 Visual intuition: What do CNN kernels detect?

Think of kernels (filters) as "templates" that detect specific patterns. Each kernel is a small matrix of weights that gets multiplied with image patches:

**Layer 1 kernels (Edge detectors)**:
These are like the first hidden layer, but each "neuron" (kernel) specializes in detecting edges:
```
Vertical edge detector:     Horizontal edge detector:    Diagonal edge detector:
[-1  0  1]                 [-1 -1 -1]                   [ 0  1  1]
[-1  0  1]                 [ 0  0  0]                   [-1  0  1]
[-1  0  1]                 [ 1  1  1]                   [-1 -1  0]
```

**Layer 2-3 kernels (Textures and patterns)**:
These combine the edge responses from layer 1 to detect more complex patterns:
- Combinations of edges form corners, curves, simple shapes
- Kernels detect wood grain, fabric textures, repeated patterns

**Layer 4-5 kernels (Object parts)**:
These are like higher hidden layers that detect object components:
- Car wheels, faces, building windows
- Spatial arrangements of lower-level features

**Final layer kernels (Complete objects)**:
Like the final hidden layer, these detect full objects:
- Complete objects: cars, faces, buildings
- Highly abstract representations

The key insight: Instead of learning arbitrary patterns like regular hidden layers, CNN layers learn hierarchical visual patterns that build from simple (edges) to complex (objects).

## 4 Formal definition of a 2-D convolution  
Let the \(q\)-th hidden volume be  
\[
H^{(q)}\in\mathbb{R}^{L_q\times B_q\times d_q},
\]  
and let the \(p\)-th kernel of that layer be  
\[
W^{(p,q)}=\bigl[w_{ijk}^{(p,q)}\bigr]_{1\le i,j\le F_q,\;1\le k\le d_q}.
\]  
With stride \(S_q=1\) and zero padding the forward map is  

\[
h^{(q+1)}_{i,j,p}\;=\;\sum_{r=1}^{F_q}\sum_{s=1}^{F_q}\sum_{k=1}^{d_q}
w^{(p,q)}_{rsk}\;h^{(q)}_{\,i+r-1,\;j+s-1,\;k},
\qquad
\begin{aligned}
&1\le i\le L_q-F_q+1,\\[-2pt]
&1\le j\le B_q-F_q+1,\\[-2pt]
&1\le p\le d_{q+1}.
\end{aligned}
\]  

This sliding-window dot product (just like the dot product in regular neural networks, but applied to small patches) is applied at **every** valid spatial location, so translating the input merely translates the feature map—**equivariance to translation**. Because the same kernel (same set of weights) is reused across all locations, the parameter count is  

\[
\#\text{weights}=F_q^{\,2}\,d_q\,d_{q+1}+d_{q+1},
\]  

independent of \(L_q,B_q\).

### Step-by-step convolution example
Think of this as applying the same "neuron" (with weights in the kernel) to every 3×3 patch in the input:

Consider a 5×5 input with a 3×3 kernel:
```
Input (like a 5×5 image):    Kernel (like neuron weights):
[1 2 3 4 5]                 [1 0 -1]
[0 1 2 3 4]                 [1 0 -1]  
[5 0 1 2 3]                 [1 0 -1]
[4 5 0 1 2]
[3 4 5 0 1]
```

**Position (1,1)**: Apply the kernel to the top-left 3×3 patch:
1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1) + 5×1 + 0×0 + 1×(-1) = **1**

**Position (1,2)**: Slide the kernel one position right:
2×1 + 3×0 + 4×(-1) + 1×1 + 2×0 + 3×(-1) + 0×1 + 1×0 + 2×(-1) = **-6**

This process creates a 3×3 feature map (output), where each value comes from applying the same kernel to different patches. It's like having 9 neurons that all share the same weights but look at different parts of the input.

## 5 Padding, stride, and receptive-field dynamics  
*Zero-padding* adds \(P\) rows/columns of zeros around the input so edge pixels can participate fully in convolution (imagine adding a black border around an image). *Full-padding* adds \(F_q-1\) zeros on every side, **increasing** the spatial footprint instead of shrinking it; full padding is vital in auto-encoders and in gradient back-propagation because it exactly inverts the shrinkage of a valid convolution.

*Stride* \(S_q\) controls how many pixels the kernel moves each step. Instead of sliding one pixel at a time, stride=2 means skip every other position. This samples the convolution at positions \(1,\;S_q+1,\;2S_q+1,\dots\) so the next layer's spatial size becomes  

\[
L_{q+1}=\left\lfloor\frac{L_q+2P-F_q}{S_q}\right\rfloor+1,\qquad
B_{q+1}=\left\lfloor\frac{B_q+2P-F_q}{S_q}\right\rfloor+1,
\]  

and each neuron's **receptive field** (the region of the original input that influences one neuron's output) grows rapidly; strides of 1 (occasionally 2) are typical because larger values degrade accuracy.

Stacking \(m\) layers of \(3\times3\) kernels with stride 1 yields an effective field  

\[
F_{\text{eff}} = 3 + 2(m-1),
\]  

so three such layers "see" a \(7\times7\) patch of the original image while using dramatically fewer parameters than a single \(7\times7\) kernel.

### Example: Receptive field growth
Consider a simple CNN with three consecutive \(3\times3\) convolutional layers (stride 1, no padding):

- **Layer 1**: Each neuron sees a \(3\times3\) patch of the input
- **Layer 2**: Each neuron aggregates information from a \(3\times3\) patch of Layer 1's output. Since each Layer 1 neuron already sees \(3\times3\), Layer 2 neurons effectively see \(3 + 2(1) = 5\times5\) of the original input
- **Layer 3**: Following the same logic, each neuron sees \(3 + 2(2) = 7\times7\) of the original input

Parameter comparison:
- Three \(3\times3\) layers: \(3 \times (3^2 \times d \times d) = 27d^2\) parameters (per channel)
- One \(7\times7\) layer: \(7^2 \times d \times d = 49d^2\) parameters (per channel)

The stacked approach uses ~45% fewer parameters while achieving the same receptive field size.

## 6 Non-linear activation and pooling  
Each convolution is immediately followed by a **ReLU** \(g(x)=\max(0,x)\) activation function (just like in regular neural networks), whose piecewise-linear derivative avoids vanishing gradients and speeds training; ReLU has almost entirely displaced sigmoid and \(\tanh\) in CNN practice.

After two or three conv-ReLU pairs, **max-pooling** is a downsampling operation that takes the maximum value from each small region (like 2×2 patches). With window \(P_q\) and stride \(S_q\), it replaces each \(P_q\times P_q\) patch by its maximum, reducing spatial resolution and imparting partial translation invariance while preserving depth (number of channels/feature maps). A canonical block  

\[
\texttt{C}\,\texttt{R}\,\texttt{C}\,\texttt{R}\,\texttt{P}
\]

is repeated several times; VGG repeats this pattern five times with \(3\times3\) filters throughout. Here C=Convolution, R=ReLU activation, P=Pooling.

### Alternative activation functions
Beyond ReLU, other activation functions used in CNNs include:
- **Sigmoid**: \(\sigma(x) = \frac{1}{1+e^{-x}}\) - same as in regular neural networks, but suffers from vanishing gradients
- **Tanh**: \(\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}\) - zero-centered version of sigmoid, but still has vanishing gradients
- **Leaky ReLU**: \(f(x) = \max(0.01x, x)\) - prevents "dead neurons" that never activate
- **ELU**: \(f(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{if } x \leq 0 \end{cases}\) - smooth version that helps with self-normalization

## 7 Classic CNN architectures

### LeNet-5 (1998)
- **Input**: 32×32 grayscale images
- **Structure**: CONV(6,5×5) → POOL(2×2) → CONV(16,5×5) → POOL(2×2) → FC(120) → FC(84) → FC(10)
- **Parameters**: ~60K
- **Innovation**: First successful CNN for digit recognition

### AlexNet (2012)
- **Input**: 224×224×3 RGB images
- **Structure**: 
  - CONV(96,11×11,stride=4) → POOL(3×3,stride=2)
  - CONV(256,5×5) → POOL(3×3,stride=2)
  - CONV(384,3×3) → CONV(384,3×3) → CONV(256,3×3) → POOL(3×3,stride=2)
  - FC(4096) → FC(4096) → FC(1000)
- **Parameters**: ~60M
- **Innovations**: ReLU, dropout, data augmentation, GPU training

### VGG-16 (2014)
- **Philosophy**: Very small (3×3) convolution filters throughout
- **Structure**: 13 conv layers + 3 FC layers
- **Key insight**: Two 3×3 convs have same receptive field as one 5×5 but fewer parameters
- **Parameters**: ~138M

### ResNet-50 (2015)
- **Innovation**: Skip connections solve vanishing gradient problem
- **Residual block**: \(H(x) = F(x) + x\) where \(F(x)\) is learned residual
- **Enables**: Networks with 50, 101, even 152 layers
- **Parameters**: ~26M (fewer than VGG despite being deeper!)

### Architecture comparison table
| Network | Year | Depth | Parameters | Top-1 Error | Key Innovation |
|---------|------|-------|------------|-------------|----------------|
| LeNet-5 | 1998 | 7 | 60K | N/A | First CNN |
| AlexNet | 2012 | 8 | 60M | 37.5% | ReLU, GPU |
| VGG-16 | 2014 | 16 | 138M | 28.1% | Small filters |
| ResNet-50 | 2015 | 50 | 26M | 23.9% | Skip connections |

## 8 Implementation: Building a CNN from scratch

### Basic CNN in PyTorch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # First conv block
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Second conv block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Third conv block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Classifier
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 4 * 4, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # First block: 32x32 -> 16x16
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Second block: 16x16 -> 8x8
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Third block: 8x8 -> 4x4
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and classify
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = self.fc(x)
        return x
```

### Training loop
```python
def train_cnn(model, train_loader, val_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        scheduler.step()
        
        print(f'Epoch {epoch}: Train Loss: {running_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {100.*correct/len(val_loader.dataset):.2f}%')
```

## 9 Training and back-propagation  
For stride 1, the gradient w.r.t. the previous layer is a convolution with the **spatially flipped** and **depth-transposed** kernel, and the forward/backward paddings satisfy  

\[
p_{\text{fwd}}+p_{\text{bwd}} = F_q-1
\]

Flattening each \(F_q\times F_q\times d_q\) patch into a vector shows that convolution is exactly the sparse matrix product \(Cf\); back-prop uses \(C^{\!\top}\), which immediately motivates **transposed (or de-)convolution** and the decoders of convolutional auto-encoders.

Because each weight is reused at every spatial location, its gradient is the **sum** of derivatives over **all** receptive fields in which it appears, so implementations must accumulate those contributions carefully.

### Example: Gradient accumulation for shared weights
This is where CNN backpropagation differs from regular neural networks. In a regular network, each weight connects to one input, so its gradient comes from one source. In CNNs, each kernel weight is **shared** across many spatial locations.

Consider a \(3\times3\) kernel applied to a \(5\times5\) input with stride 1. The kernel weight \(w_{1,1}\) (top-left corner) participates in \(3\times3 = 9\) different convolution operations across the output (because the kernel slides to 9 different positions). During backpropagation, the gradient for \(w_{1,1}\) must be the **sum** of gradients from all 9 locations where it was used:

\[
\frac{\partial L}{\partial w_{1,1}} = \sum_{i=1}^{3}\sum_{j=1}^{3} \frac{\partial L}{\partial h_{i,j}} \cdot x_{i,j}
\]

where \(h_{i,j}\) are the output activations and \(x_{i,j}\) are the corresponding input values that multiplied \(w_{1,1}\).

This weight sharing is why CNNs can learn translation-invariant features: the same edge detector (kernel) learns to detect edges everywhere in the image, not just in one specific location.

## 10 Hyperparameter tuning and optimization

### Critical hyperparameters
- **Learning rate**: Start with 0.001 for Adam, 0.1 for SGD
- **Batch size**: 32-128 typical, larger for better GPU utilization
- **Filter sizes**: 3×3 most common, occasional 5×5 or 7×7 in first layer
- **Number of filters**: Double after each pooling layer (32→64→128→256)
- **Dropout**: 0.5 in fully connected layers, 0.2-0.3 in conv layers

### Regularization techniques
These help prevent overfitting (when the network memorizes training data instead of learning general patterns):
- **Batch normalization**: Normalizes inputs to each layer, making training more stable and faster
- **Dropout**: Randomly sets some neurons to zero during training to prevent over-reliance on specific features
- **L2 regularization**: Adds a penalty \(\lambda\sum w_i^2\) to the loss function to keep weights small
- **Data augmentation**: Creates new training examples by rotating, flipping, or slightly modifying existing images

### Optimization strategies
- **Adam**: Adaptive learning rates, good default choice
- **SGD with momentum**: Often better final performance with proper tuning
- **Learning rate scheduling**: Reduce LR when validation loss plateaus
- **Warm restarts**: Periodically reset learning rate to escape local minima

## 11 Common failure modes and debugging

### Vanishing gradients
- **Symptoms**: Training loss stops decreasing, gradients approach zero
- **Solutions**: Skip connections (ResNet), batch normalization, better initialization

### Overfitting
- **Symptoms**: Training accuracy high, validation accuracy low
- **Solutions**: More data, dropout, regularization, early stopping

### Underfitting
- **Symptoms**: Both training and validation accuracy low
- **Solutions**: Larger model, lower regularization, longer training

### Dead ReLU neurons
- **Symptoms**: Many neurons always output zero
- **Solutions**: Lower learning rate, Leaky ReLU, better initialization

### Debugging checklist
1. **Start simple**: Single layer CNN first
2. **Check data**: Visualize inputs, verify labels
3. **Monitor gradients**: Use gradient clipping if exploding
4. **Learning rate**: Too high causes instability, too low causes slow convergence
5. **Batch size**: Very small causes noisy gradients, very large causes poor generalization

## 12 Transfer learning and practical applications

### Transfer learning workflow
**Transfer learning** lets you use a pre-trained network (trained on millions of images) and adapt it for your specific task:
1. **Choose pretrained model**: ResNet, VGG, EfficientNet trained on ImageNet (1.2M images, 1000 classes)
2. **Remove final layer**: Replace the 1000-class classifier with your task-specific classifier (e.g., 2 classes for cat/dog)
3. **Freeze early layers**: Keep the pre-trained feature extractors (edge detectors, texture detectors) fixed initially
4. **Fine-tune gradually**: Slowly unfreeze layers from top to bottom, allowing them to adapt to your specific data

### Example: Transfer learning for medical imaging
```python
import torchvision.models as models

# Load pretrained ResNet
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for binary classification
model.fc = nn.Linear(model.fc.in_features, 2)

# Only train the final layer initially
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### When to use transfer learning
- **Small dataset**: Almost always beneficial
- **Similar domain**: Natural images → other natural images
- **Different domain**: May need more layers unfrozen
- **Sufficient data**: Training from scratch might be better

## 13 Design heuristics and capacity control  
Typical choices are square inputs (\(L_q=B_q\)), powers-of-two channel counts, and small filters (\(F_q\in\{3,5\}\)). Small filters permit greater depth for a fixed parameter budget; VGG's decision to use \(3\times3\) throughout achieved state-of-the-art ImageNet accuracy with only 15 weight layers.

Increasing the number of filters in layer \(q\) directly increases the depth \(d_{q+1}\) of its output, expanding model capacity; late layers therefore tend to be shallow in space but very deep in *channels* (hundreds) to capture diverse high-level concepts.

Residual and densely connected blocks further ease optimization in models exceeding 100 layers, while *strided convolutions* increasingly replace pooling to keep everything differentiable and to grow receptive fields faster.

Skip connections, batch normalization, data augmentation, and adaptive optimizers (Adam, AdaBelief) complete the training toolbox, but the core algebra—convolution, ReLU, pooling, transposed convolution—remains unchanged since LeNet-5.

## 14 Computational considerations

### Memory usage
- **Activations**: Dominate memory usage, scale with batch size
- **Weights**: Fixed cost, shared across spatial locations
- **Gradients**: Same size as weights during backprop

### Speed optimization
- **Convolution implementations**: im2col, Winograd, FFT-based
- **Mixed precision**: Use 16-bit floats for forward pass, 32-bit for gradients
- **Batch processing**: Vectorize operations across samples
- **GPU utilization**: Keep GPU busy with proper batch sizes

### Model compression
- **Pruning**: Remove less important weights
- **Quantization**: Use 8-bit instead of 32-bit weights
- **Knowledge distillation**: Train smaller model to mimic larger one
- **MobileNets**: Depthwise separable convolutions for mobile deployment

---

# One-dimensional convolutional networks

## 15 Sequences as one-dimensional grids  
Text, time-series, and other ordered data can be regarded as *1-D grids* where the "spatial" axis becomes time and the **depth** becomes the feature dimension. For example, a word embedding turns each word into a vector of numbers, so a sentence becomes a matrix where each row is a word's vector.

**TextCNN** treats a sentence of length \(T\) as a matrix in \(\mathbb{R}^{T\times d}\) (T words, each with d-dimensional embeddings) and slides kernels of width \(w\) across all contiguous \(w\)-grams (sequences of w words).

A kernel therefore acts like a detector for a particular \(w\)-word phrase or pattern; kernels of multiple widths (like 3, 4, 5 words) capture short phrases, clauses, or longer dependencies. The convolution operation at every temporal location shares weights across time, so the same phrase detector fires wherever that phrase appears—an exact 1-D analogue of how image kernels detect the same visual pattern everywhere in an image.

### Example: Sentiment analysis with TextCNN
```python
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))  # (batch, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        x = torch.cat(conv_outputs, dim=1)  # Concatenate all filter outputs
        x = self.dropout(x)
        return self.fc(x)
```

## 16 Mathematics inherited unchanged  
Collapsing the breadth dimension reduces the tensor indices from \((i,j,k)\) to \((t,k)\), but **all formulas from earlier sections still hold**. A 1-D layer with kernel width \(F_q\) and depths \(d_q,d_{q+1}\) has  

\[
F_q\,d_q\,d_{q+1}+d_{q+1}
\]

trainable parameters; padding, stride, and global max-pooling regulate temporal footprint exactly as in 2-D. Back-propagation again uses an inverted kernel and, for stride 1, obeys \(p_{\text{fwd}}+p_{\text{bwd}}=F_q-1\).

## 17 Applications of 1-D CNNs

### Time series forecasting
- **Input**: Historical sensor readings, stock prices, weather data
- **Architecture**: Multiple conv layers with dilated convolutions for long-range dependencies
- **Advantage**: Parallelizable, faster than RNNs for long sequences

### Audio processing
- **Input**: Raw waveforms or spectrograms
- **Architecture**: Deep 1-D CNNs with small kernels (3-7 samples)
- **Applications**: Speech recognition, music classification, audio event detection

### Genomics
- **Input**: DNA/protein sequences as categorical data
- **Architecture**: Multiple filter sizes to capture motifs of different lengths
- **Applications**: Gene expression prediction, protein function classification

---

# Temporal Convolutional Networks (TCNs)

## 18 Beyond standard 1-D CNNs: The temporal challenge

While standard 1-D CNNs (like the ones we just discussed) excel at detecting local patterns in sequences, they face fundamental limitations for modeling long-term temporal dependencies:

1. **Limited receptive field**: To "see" 100 time steps back, you'd need many layers, making the network very deep
2. **Causality**: In real-time applications, future information shouldn't influence past predictions (you can't use tomorrow's stock price to predict today's)
3. **Variable-length sequences**: Standard CNNs produce fixed-size outputs, but sequences can have different lengths
4. **Computational efficiency**: RNNs (Recurrent Neural Networks) process sequences one step at a time, while CNNs can process all steps in parallel

**Temporal Convolutional Networks (TCNs)** solve these challenges through two key innovations:
- **Dilated convolutions**: Create gaps in the kernel to "see" farther back in time without adding parameters
- **Causal convolutions**: Ensure the output at time t only depends on inputs at time t and earlier, never future inputs

## 19 Dilated convolutions: Exponential receptive field growth

A **dilated convolution** (also called **atrous convolution**) is like a regular convolution, but with gaps between the kernel elements. Instead of looking at consecutive time steps, it skips some steps, controlled by the dilation rate \(d\):

\[
y_i = \sum_{k=0}^{K-1} w_k \cdot x_{i-k \cdot d}
\]

where \(K\) is the kernel size, \(d\) is the dilation rate, and \(w_k\) are the kernel weights (just like in regular convolution).

Think of it as "stretching" the kernel: instead of looking at 3 consecutive time steps, a dilated kernel with rate 2 looks at every other time step, effectively seeing 5 time steps of history but with the same number of parameters.

### Visual representation
```
Standard 3-point convolution (d=1):
Input:  [..., x_{i-2}, x_{i-1}, x_i, x_{i+1}, x_{i+2}, ...]
Kernel:              [w_0,   w_1,  w_2]

Dilated convolution (d=2):
Input:  [..., x_{i-4}, x_{i-3}, x_{i-2}, x_{i-1}, x_i, x_{i+1}, x_{i+2}, ...]
Kernel:              [w_0,           w_1,           w_2]
```

### Exponential receptive field growth
Stacking dilated convolutions with exponentially increasing dilation rates creates exponential receptive field growth:

- **Layer 1**: dilation=1, receptive field = 3
- **Layer 2**: dilation=2, receptive field = 7  
- **Layer 3**: dilation=4, receptive field = 15
- **Layer 4**: dilation=8, receptive field = 31

General formula for \(L\) layers with kernel size \(K\) and dilation rates \(d_1, d_2, ..., d_L\):
\[
\text{Receptive field} = 1 + \sum_{i=1}^{L} (K-1) \cdot d_i
\]

For exponential dilation (\(d_i = 2^{i-1}\)) with \(K=3\):
\[
\text{Receptive field} = 1 + 2 \sum_{i=0}^{L-1} 2^i = 1 + 2(2^L - 1) = 2^{L+1} - 1
\]

## 20 Causal convolutions: Respecting temporal order

**Causal convolutions** ensure that the output at time \(t\) depends only on inputs at times \(t\) and earlier, never on future inputs. This is crucial for real-time applications where you can't "look into the future."

In regular convolution, a 3-element kernel centered at time t would use inputs from t-1, t, and t+1. In causal convolution, the kernel is shifted so it only uses inputs from t-2, t-1, and t.

### Implementation through padding
For a kernel of size \(K\) and dilation \(d\), causal convolution requires **left-padding** (adding zeros to the left) of size \((K-1) \cdot d\). This ensures the convolution only "looks backward" in time:

```python
def causal_conv1d(x, weight, dilation=1):
    # x shape: (batch, channels, seq_len)
    kernel_size = weight.size(-1)
    padding = (kernel_size - 1) * dilation
    
    # Left-pad the input
    x_padded = F.pad(x, (padding, 0))
    
    # Apply dilated convolution
    out = F.conv1d(x_padded, weight, dilation=dilation)
    
    # Truncate to original length
    return out[:, :, :x.size(-1)]
```

### Causal vs. non-causal comparison
```
Non-causal (standard) convolution:
t: ... t-2  t-1   t   t+1  t+2 ...
   ... [w0  w1   w2]           ... (output at t uses t-1, t, t+1)

Causal convolution:
t: ... t-2  t-1   t   t+1  t+2 ...
   ... [w0  w1   w2]           ... (output at t uses t-2, t-1, t)
```

## 21 TCN architecture and residual connections

A complete TCN combines dilated causal convolutions with residual connections to enable very deep networks:

### Basic TCN block
```python
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super(TCNBlock, self).__init__()
        
        # First dilated causal conv
        self.conv1 = self._causal_conv(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated causal conv
        self.conv2 = self._causal_conv(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def _causal_conv(self, in_channels, out_channels, kernel_size, dilation):
        padding = (kernel_size - 1) * dilation
        return nn.Conv1d(in_channels, out_channels, kernel_size, 
                        padding=padding, dilation=dilation)
    
    def forward(self, x):
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :x.size(-1)]  # Causal truncation
        out = self.dropout1(F.relu(self.norm1(out)))
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :x.size(-1)]  # Causal truncation
        out = self.dropout2(F.relu(self.norm2(out)))
        
        # Residual connection
        return F.relu(out + self.residual(x))
```

### Complete TCN architecture
```python
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, 
                                 dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Example usage
tcn = TemporalConvNet(num_inputs=1, num_channels=[32, 32, 32, 32], kernel_size=3)
# This creates a 4-layer TCN with receptive field of 31 time steps
```

## 22 TCN vs. RNN comparison

### Computational advantages
- **Parallelization**: All time steps processed simultaneously
- **Gradient flow**: No vanishing gradient problem through time
- **Memory efficiency**: Constant memory usage vs. RNN's linear growth
- **Training speed**: Typically 2-3x faster than RNNs

### Memory usage comparison
```python
# RNN memory usage (sequential)
def rnn_memory_usage(seq_len, hidden_size, batch_size):
    return seq_len * hidden_size * batch_size  # Linear in sequence length

# TCN memory usage (parallel)
def tcn_memory_usage(seq_len, num_channels, batch_size):
    return seq_len * max(num_channels) * batch_size  # Constant per layer
```

### Performance comparison table
| Aspect | RNN/LSTM | TCN |
|--------|----------|-----|
| Training Speed | Slow (sequential) | Fast (parallel) |
| Memory Usage | O(seq_len) | O(1) per layer |
| Gradient Flow | Vanishing gradients | Stable |
| Receptive Field | Unlimited | Limited but tunable |
| Causality | Natural | Enforced by design |

## 23 Applications and use cases

### Financial time series prediction
```python
class StockPredictor(nn.Module):
    def __init__(self, input_features=5, hidden_channels=[32, 64, 128, 64]):
        super(StockPredictor, self).__init__()
        self.tcn = TemporalConvNet(input_features, hidden_channels)
        self.classifier = nn.Linear(hidden_channels[-1], 1)
        
    def forward(self, x):
        # x shape: (batch, features, seq_len)
        tcn_out = self.tcn(x)
        # Use last time step for prediction
        return self.classifier(tcn_out[:, :, -1])
```

### Audio generation (WaveNet-style)
```python
class WaveNetTCN(nn.Module):
    def __init__(self, num_classes=256, num_layers=10, channels=32):
        super(WaveNetTCN, self).__init__()
        
        # Exponential dilation pattern
        dilations = [2**i for i in range(num_layers)]
        
        self.layers = nn.ModuleList([
            TCNBlock(1 if i == 0 else channels, channels, 
                    kernel_size=2, dilation=dilations[i])
            for i in range(num_layers)
        ])
        
        self.output = nn.Conv1d(channels, num_classes, 1)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
```

### Real-time streaming applications
TCNs excel in streaming scenarios where:
- **Low latency** is crucial (no need to wait for full sequence)
- **Fixed computational budget** per time step
- **Causal processing** is required (no future information)

Example: Real-time speech recognition, live audio processing, high-frequency trading

## 24 Advanced TCN variants

### Gated TCN
Incorporates gating mechanisms similar to LSTM:
```python
class GatedTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(GatedTCNBlock, self).__init__()
        
        # Parallel convolutions for filter and gate
        self.conv_filter = self._causal_conv(in_channels, out_channels, kernel_size, dilation)
        self.conv_gate = self._causal_conv(in_channels, out_channels, kernel_size, dilation)
        
    def forward(self, x):
        filter_out = torch.tanh(self.conv_filter(x))
        gate_out = torch.sigmoid(self.conv_gate(x))
        return filter_out * gate_out  # Gated activation
```

### Multi-scale TCN
Uses multiple dilation rates in parallel:
```python
class MultiScaleTCN(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4, 8]):
        super(MultiScaleTCN, self).__init__()
        
        self.branches = nn.ModuleList([
            TCNBlock(in_channels, out_channels//len(dilations), 3, d)
            for d in dilations
        ])
        
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        return torch.cat(branch_outputs, dim=1)  # Concatenate along channel dimension
```

## 25 Practical considerations and limitations

### Advantages
- **Faster training**: Parallelizable across time steps
- **Stable gradients**: No vanishing gradient through time
- **Flexible receptive fields**: Tunable via dilation pattern
- **Memory efficient**: Constant memory per layer
- **Deterministic**: Same input always produces same output

### Limitations
- **Fixed receptive field**: Cannot adapt to variable-length dependencies
- **Memory vs. receptive field tradeoff**: Larger fields require more layers
- **Less interpretable**: Harder to understand what the network "remembers"
- **Padding artifacts**: Causal padding can introduce boundary effects

### Design guidelines
1. **Receptive field sizing**: Ensure field covers longest relevant dependency
2. **Dilation pattern**: Exponential (1,2,4,8,...) is most common
3. **Kernel size**: 3 is typical, 2 for WaveNet-style generation
4. **Depth vs. width**: Deeper networks (more layers) vs. wider (more channels)
5. **Regularization**: Dropout and batch normalization are crucial

---

# Concluding synthesis

Convolutional networks form a single, mathematically coherent family whose core operation is a sparse, shared-parameter dot product that is equivariant to translation. The algebra of that operation—matrix \(C\) versus its transpose \(C^{\!\top}\)—governs feature extraction, gradient flow, transposed convolution, and even the decoders of auto-encoders. Whether the grid is two-dimensional (images), one-dimensional (sentences, biosignals), or three-dimensional (video), the same design axioms apply:

* **Locality** controls parameters and embeds domain knowledge  
* **Weight sharing** imposes translational consistency  
* **Hierarchical depth** builds complicated concepts from primitive ones  
* **Non-linear activations** and **pooling/striding** grow receptive fields while retaining computational efficiency  

Armed with these principles, the mathematical formulations, implementation examples, and practical debugging strategies above, a practitioner can implement, train, and deploy CNNs across diverse applications—from image classification to genomics—with complete theoretical understanding and practical confidence.

The field continues to evolve with attention mechanisms, transformers, and hybrid architectures, but the convolutional foundation remains essential for understanding how neural networks can efficiently process structured data with spatial or temporal relationships.





# Next Steps Implementation Plan

## Improve MLN

### Current State
The model implements a basic 3-layer multilayer perceptron with 4242 input features, hidden layers of 6000 and 4000 neurons, and 303 outputs for regression. It uses ReLU activations, 50% dropout after the first layer, and NAdam optimizer with a fixed learning rate of 0.0005. The model trains for exactly 1000 epochs every time across 5-fold cross-validation.

### Changes
1. **Add early stopping**
  1. monitoring validation loss and stopping training when it doesn't improve for 15-20 epochs
  2. Prevents overfitting by halting training before the model memorizes the training data
2. **Implement learning rate scheduling using ReduceLROnPlateau**
  1. a technique that automatically reduces the learning rate by a factor (like dividing by 2) when validation loss stops improving
  2. Allows the model to make smaller, more precise weight updates later in training
3. **Add explicit He weight initialization**
  1. **He weight initialization uses a normal distribution with mean 0 and standard deviation of sqrt(2/n_in), where n_in is the number of input neurons to that layer.** This specific mathematical formula is derived to maintain proper variance of activations and gradients as they flow through ReLU layers, with the sqrt(2) factor accounting for ReLU setting negative values to zero. It prevents gradients from vanishing (becoming extremely small, causing early layers to barely learn) or exploding (becoming extremely large, causing unstable training with huge weight updates) at the start of training.
4. **Include batch normalization layers**
  1. normalize the outputs from each layer before they go into the next layer - even though you normalized the original input data, as it passes through layers with weights and activations, the data distribution changes and can become skewed again, so batch normalization re-normalizes at each layer to keep the data well-behaved
5. **Implement gradient clipping**
  1. a technique that prevents gradients from becoming too large by capping them at a maximum value (like 1.0)
  2. stops the weights from making huge jumps that could destabilize training

### Expected Benefits
These changes would reduce training time through early stopping, improve generalization performance by 5-15%, and make training more stable and consistent across folds.

## Implement 1D CNN

### Rationale for 1D CNN
IMU sensor data has inherent temporal structure - accelerometer and gyroscope readings are sequences where nearby time points are related and contain patterns over time. MLNs treat each time point independently, ignoring this temporal structure. 1D CNNs can detect local temporal patterns (like specific motion signatures) and learn translation-invariant features that recognize the same motion pattern regardless of when it occurs in the sequence.

### Architecture Design
Replace the current MLN with a 1D CNN that treats the 4242 input features as a temporal sequence. The architecture would consist of multiple 1D convolutional layers with different kernel sizes (3, 5, 7) to capture patterns at different time scales, followed by max pooling layers to reduce dimensionality and create translation invariance. The final convolutional layers would be followed by global average pooling and fully connected layers to produce the 303 outputs.

### Key Components
1. **Convolutional layers**: Use 1D convolutions with kernel sizes 3-7 to detect local temporal patterns in the IMU data
2. **Multiple filter banks**: Apply 32-128 filters per layer to learn diverse temporal features
3. **Max pooling**: Reduce temporal resolution while maintaining important features
4. **Residual connections**: Add skip connections to enable deeper networks and better gradient flow
5. **Global pooling**: Replace flatten with global average pooling to handle variable sequence lengths

### Implementation Details
The input would be reshaped from (batch_size, 4242) to (batch_size, channels, sequence_length) where channels represent different sensor axes (x, y, z for accelerometer and gyroscope) and sequence_length represents time steps. Multiple parallel branches with different kernel sizes would capture short-term and long-term temporal dependencies, with outputs concatenated before final prediction layers.

## Implement Temporal CNN (TCN)

### Rationale for TCN
While 1D CNNs can capture local temporal patterns, they struggle with long-range dependencies in sequential data. TCNs address this through dilated convolutions that exponentially expand the receptive field, allowing the network to "see" much further back in time without dramatically increasing parameters. For IMU data predicting movement outcomes, long-range dependencies are crucial as current motion depends on motion patterns from many time steps ago.

### Architecture Design
Implement a TCN with exponentially increasing dilation rates (1, 2, 4, 8, 16) to create a receptive field covering the entire input sequence. Each TCN block would contain two dilated causal convolutions with residual connections. The causal nature ensures that predictions only use past and present information, not future data, which is important for real-time applications.

### Key Components
1. **Dilated convolutions**: Use dilation rates of 1, 2, 4, 8, 16 to create exponential receptive field growth
2. **Causal convolutions**: Ensure output at time t only depends on inputs at times ≤ t through proper padding
3. **Residual blocks**: Each TCN block contains two dilated convolutions with skip connections
4. **Exponential receptive field**: 5 layers with dilation pattern allows seeing ~31 time steps back
5. **Weight sharing**: Same temporal pattern detectors applied across all time positions

### Implementation Details
Each TCN block would have two 1D convolutional layers with the same dilation rate, batch normalization, ReLU activation, and dropout. The blocks would be stacked with increasing dilation rates. Causal padding would be implemented by padding only the left side of the input to ensure future information doesn't leak into past predictions. The final output would use the last time step's representation for regression.

## Comparison Strategy
Train all three models (improved MLN, 1D CNN, TCN) on the same cross-validation splits and compare performance using the same nRMSE metrics. The MLN serves as a baseline, the 1D CNN tests whether local temporal patterns improve prediction, and the TCN tests whether long-range temporal dependencies are crucial for the task. This systematic comparison will reveal which architectural approach best captures the underlying patterns in the IMU data for predicting movement outcomes.
