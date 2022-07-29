import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def bi_cross_entropy(output, target):
  output_1, output_2 = output[:, :, 0], output[:, :, 1]
  target_1, target_2 = target[:, 0], target[:, 1]
  ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")
  return ce_loss(output_1, target_1) + ce_loss(output_2, target_2)