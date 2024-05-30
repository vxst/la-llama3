import torch


def apply_rope(input_tensor, rope_theta=500000.0, idx=0):
    input_tensor = input_tensor.view(input_tensor.shape[0], input_tensor.shape[1]//2, 2)
    output_tensor = torch.zeros_like(input_tensor)
    
    seq_len = input_tensor.shape[0]
    dim = input_tensor.shape[1]
    
    position_ids = torch.arange(128, dtype=torch.float32).unsqueeze(1)  # Shape: [seq_len, 1]
    
    theta = rope_theta
    dim_indices = torch.arange(dim, dtype=torch.float32)
    inv_freq = 1.0 / (theta ** (dim_indices / dim))
    angles = position_ids * inv_freq
    
    angles_cos = torch.cos(angles)
    angles_sin = torch.sin(angles)

    for i in range(seq_len):
        for j in range(dim):
            real_part = input_tensor[i, j, 0]
            imag_part = input_tensor[i, j, 1]
            cos_angle = angles_cos[i+idx, j]
            sin_angle = angles_sin[i+idx, j]

            output_tensor[i, j, 0] = real_part * cos_angle - imag_part * sin_angle
            output_tensor[i, j, 1] = real_part * sin_angle + imag_part * cos_angle

    return output_tensor.view(input_tensor.shape[0], input_tensor.shape[1]*2)