"""
Targeted Knowledge Distillation based on Corrective Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stores a list of middle-tier outputs
teacher_last = []
student_last = []
gradients = []


def enhance_weights(weights):
    # The weights are first normalized by softmax
    normalized_weights = F.softmax(weights, dim = 1)

    # Enhancement with Exponential Functions
    enhanced_weights = torch.exp(normalized_weights * 2.0)

    # normalize again
    adjusted_weights = F.softmax(enhanced_weights, dim = 1)

    return adjusted_weights


def MixAttention(feature_maps):
    # 1. Calculate the mean and standard deviation for each channel
    channel_mean = torch.mean(feature_maps, dim = (2, 3), keepdim = True)  # (B, C, 1, 1)
    channel_std = torch.std(feature_maps, dim = (2, 3), keepdim = True)  # (B, C, 1, 1)

    # 2. Calculate channel importance scores based on mean and standard deviation
    # Channels with large standard deviation contain more discriminatory information
    channel_weights = channel_std / (channel_mean + 1e-5)  # 避免除零

    # 3. Normalize the weights to the [0,1] interval
    channel_weights = torch.sigmoid(channel_weights)

    # 1. Calculating the activation strength of a spatial dimension
    spatial_weights = torch.mean(torch.abs(feature_maps), dim = 1, keepdim = True)  # (B, 1, H, W)

    # 2. normalize
    spatial_weights = F.normalize(spatial_weights, p = 2, dim = (2, 3), eps = 1e-12)

    # 3. Apply Softmax to make the weights sum to 1
    spatial_weights = F.softmax(spatial_weights.view(*spatial_weights.shape[:2], -1), dim = 2)
    spatial_weights = spatial_weights.view_as(torch.mean(feature_maps, dim = 1, keepdim = True))

    # 4. Application weights
    weighted_features = feature_maps * channel_weights
    final_weighted = weighted_features * spatial_weights

    return final_weighted


# Defining Hook Functions
def teacher_hook(module, input, output):
    teacher_last.append(output)


def student_hook(module, input, output):
    student_last.append(output)


def save_gradients(module, grad_input, grad_output):
    gradients.append(grad_output[0])


class Loss_compute:
    def __init__(self):
        self.beta = 3.0     # Balancing the weighting of soft and hard targets
        self.T = 3.0

    def __call__(self, teacher, student, images, labels, device):
        # Registering hook functions at the middle layer of the teacher network and the student network
        # layer4[-1]/features.denseblock4.denselayer16

        teacher.features.denseblock4.denselayer16.register_forward_hook(teacher_hook)
        student.features[10].denselayer16.conv2.register_forward_hook(student_hook)
        teacher.features.denseblock4.denselayer16.conv2.register_full_backward_hook(save_gradients)

        # teacher.layer4[-1].register_forward_hook(teacher_hook)
        # student.layer4[-1].register_forward_hook(student_hook)
        # teacher.layer4[-1].register_full_backward_hook(save_gradients)

        # forward propagation
        teacher.eval()  # Teachers' networks are not updated on a gradient
        teacher_output = teacher(images)
        student_output = student(images)
        # Assume that the target classification result is the maximum probability category for each sample
        target_classes = torch.argmax(teacher_output, dim = 1)
        # Creating a one-hot code for calculating the gradient
        one_hot = torch.zeros(teacher_output.size(), device = device)
        one_hot.scatter_(1, target_classes.unsqueeze(1), 1)
        # Ensure features tensor preserves gradients
        teacher_last[0].retain_grad()
        # Calculating the gradient
        teacher_output.backward(gradient = one_hot, retain_graph = True)
        # Getting the gradient
        weights = gradients[0]
        weights = torch.mean(weights, dim = (2, 3))
        weights = enhance_weights(weights)
        teacher_last[0] = teacher_last[0] * weights.unsqueeze(2).unsqueeze(3)
        teacher_last[0] = MixAttention(teacher_last[0])
        # Calculating Attention Correction Losses
        cos_sim = F.cosine_similarity(student_last[0], teacher_last[0], dim = 1)
        att_loss = 1 - cos_sim.mean()
        # Empty the list of middle-tier outputs
        teacher_last.clear()
        student_last.clear()
        gradients.clear()

        # Define a hard loss function, using cross-entropy loss
        criterion_ce = nn.CrossEntropyLoss()
        hard_loss = criterion_ce(student_output, labels)

        # # 定义软损失函数，这里使用交叉熵损失和 KL 散度损失
        # criterion_kl = nn.KLDivLoss(reduction = 'sum')
        # soft_loss = criterion_kl(
        #     torch.nn.functional.log_softmax(student_output / self.T, dim = 1),
        #     # torch.nn.functional.softmax(teacher_output / self.T, dim=1)
        #     torch.nn.functional.softmax(teacher_output, dim = 1)
        # )
        loss_tal = hard_loss + self.beta * att_loss

        return loss_tal
