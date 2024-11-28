import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16, VGG16_Weights
from torchvision import datasets, transforms


# Define Tensor Reorganization
def tensor_reorganization(tensor):
    if tensor.dim() == 4:  # For convolutional weights (4D tensors)
        return tensor.permute(1, 0, 2, 3).mean(dim=1, keepdim=True)
    elif tensor.dim() == 3:  # For FC layer weights with batch dimension (3D tensors)
        return tensor.mean(dim=0, keepdim=True).unsqueeze(0)
    elif tensor.dim() == 2:  # For FC layer weights without batch dimension
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 1:  # For FC layer biases
        return tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}. Only 2D, 3D, and 4D tensors are supported.")


# Define Predictor Model
class PredictorModel(nn.Module):
    def __init__(self, hidden_channels, pool_output_size, fixed_output_size):
        super(PredictorModel, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(pool_output_size)
        self.conv1 = nn.Conv2d(1, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        flattened_size = hidden_channels * pool_output_size[0] * pool_output_size[1]
        self.fc = nn.Linear(flattened_size, fixed_output_size)

    def forward(self, x, output_size):
        x = self.pool(x)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.flatten(x, 1)
        pred_chunk = self.fc(x)
        return torch.mean(pred_chunk, dim=0, keepdim=True)


# Predict and Fill Gradients
def predict_and_fill(predictor, activations, output_size, fixed_output_size=1000):
    pred_chunk = predictor(activations, torch.Size([fixed_output_size])).flatten()
    total_elements = output_size.numel()
    num_full_repeats = total_elements // fixed_output_size
    remainder = total_elements % fixed_output_size
    if num_full_repeats == 0:
        return pred_chunk[:total_elements].view(*output_size)
    repeated_prediction = torch.cat([pred_chunk] * num_full_repeats)
    if remainder > 0:
        remainder_fill = pred_chunk[:remainder]
        full_prediction = torch.cat((repeated_prediction, remainder_fill))
    else:
        full_prediction = repeated_prediction
    return full_prediction.view(*output_size)


# Activation Hook
def activation_hook(module, input, output, activation_dict, layer_name):
    if isinstance(output, torch.Tensor):
        activation_dict[layer_name] = tensor_reorganization(output.detach())


from functools import partial


# Register Hooks
def register_hooks(model, activation_dict):
    for name, module in model.named_modules():
        if hasattr(module, "weight") and module.weight.requires_grad:
            hook_fn = partial(activation_hook, activation_dict=activation_dict, layer_name=name)
            module.register_forward_hook(hook_fn)


# Train ADA-GP
def train_adagp(model, predictor, dataloader, num_epochs=10, warmup_epochs=3, lr=0.001, fixed_output_size=1000, device="cuda"):
    model.to(device)
    predictor.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_model = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer_predictor = optim.Adam(predictor.parameters(), lr=lr * 0.1)
    activation_dict = {}
    register_hooks(model, activation_dict)

    for epoch in range(num_epochs):
        phase = "Warm-Up" if epoch < warmup_epochs else "Gradient Prediction"
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Phase: {phase}")
        model.train()
        predictor.train() if phase == "Warm-Up" else predictor.eval()
        epoch_loss, correct_predictions, total_samples = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer_model.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            if phase == "Warm-Up":
                if batch_idx == 0:
                    print(f"Executing Gradient Training in Warm-Up Phase for Epoch {epoch + 1}")

                loss.backward()
                optimizer_model.step()
                optimizer_predictor.zero_grad()

                predictor_loss_sum = 0
                count = 0

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_key = '.'.join(name.split('.')[:-1])
                        if param_key in activation_dict:
                            activations = activation_dict[param_key]
                            pred_grads = predict_and_fill(predictor, activations, param.grad.size(), fixed_output_size)
                            predictor_loss = nn.MSELoss()(pred_grads, param.grad.detach())
                            predictor_loss_sum += predictor_loss.item()
                            count += 1

                            predictor_loss.backward()

                optimizer_predictor.step()

                if (batch_idx + 1) % 10 == 0:
                    avg_predictor_loss = predictor_loss_sum / count if count > 0 else 0.0
                    print(f"[Warm-Up Epoch {epoch + 1}, Iter {batch_idx + 1}] Main Model Loss: {loss.item():.4f}, Avg Predictor Loss: {avg_predictor_loss:.4f}")

            elif phase == "Gradient Prediction":
                if batch_idx == 0:
                    print(f"Executing Gradient Prediction Phase for Epoch {epoch + 1}")
                for name, module in model.named_children():
                    if hasattr(module, "weight") and module.weight.requires_grad:
                        activations = tensor_reorganization(inputs)
                        pred_grads = predict_and_fill(
                            predictor=predictor,
                            activations=activations,
                            output_size=module.weight.size(),
                            fixed_output_size=fixed_output_size
                        )
                        module.weight.grad = pred_grads
                optimizer_model.step()

            # Accuracy tracking
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_accuracy = 100.0 * correct_predictions / total_samples
        print(f"Epoch [{epoch + 1}/{num_epochs}] | Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")


# Load VGG16
vgg = vgg16(weights=None)
vgg.classifier[6] = nn.Linear(vgg.classifier[6].in_features, 10)

# Initialize Predictor
predictor = PredictorModel(pool_output_size=(4, 4), hidden_channels=64, fixed_output_size=1000)

# CIFAR-10 Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

# Train ADA-GP model
train_adagp(vgg, predictor, trainloader, num_epochs=10, warmup_epochs=3, device="cuda" if torch.cuda.is_available() else "cpu")
