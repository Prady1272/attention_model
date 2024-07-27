import torch

def point_loss_function(predicted_rotation_point_rot,g_rotation_point_rot,g_axis_rot):
    cross_product = torch.cross(g_axis_rot,g_rotation_point_rot-predicted_rotation_point_rot,dim=1)
    distance = torch.linalg.norm(cross_product,dim=1)/torch.linalg.norm(g_axis_rot,dim=1)
    return torch.mean(distance**2)



test_cases = {
    "predicted_rotation_point_rot": torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [2.0, 2.0, 100.0]]),
    "g_rotation_point_rot": torch.tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 0.0]]),
    "g_axis_rot": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
}


# Test the function
calculated_losses = point_loss_function(test_cases["predicted_rotation_point_rot"],
                                        test_cases["g_rotation_point_rot"],
                                        test_cases["g_axis_rot"])

print(f"Calculated Losses: {calculated_losses}")
