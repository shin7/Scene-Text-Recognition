import torch


def dice_coefficient(y_true_cls, y_pred_cls, training_mask):
    """
    dice loss
    :param y_true_cls:
    :param y_pred_cls:
    :param training_mask:
    :return:
    """
    eps = 1e-5
    y_true_cls = y_true_cls.view(-1)
    y_pred_cls = y_pred_cls.view(-1)
    training_mask = training_mask.view(-1)

    intersection = torch.sum(y_true_cls * y_pred_cls * training_mask)
    union = torch.sum(y_true_cls * training_mask) + torch.sum(y_pred_cls * training_mask) + eps
    return 1. - (2 * intersection / union)


def loss(y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
    """
    define the loss used for training, contraning two part,
    the first part we use dice loss instead of weighted logloss,
    the second part is the iou loss defined in the paper
    :param y_true_cls: ground truth of text
    :param y_pred_cls: prediction os text
    :param y_true_geo: ground truth of geometry
    :param y_pred_geo: prediction of geometry
    :param training_mask: mask used in training, to ignore some text annotated by ###
    :return:
    """
    classification_loss = dice_coefficient(y_true_cls, y_pred_cls, training_mask)
    # scale classification loss to match the iou loss part
    classification_loss *= 0.01

    # d1 -> top, d2->right, d3->bottom, d4->left
    d1_gt = y_true_geo[:, :, :, 0]
    d2_gt = y_true_geo[:, :, :, 1]
    d3_gt = y_true_geo[:, :, :, 2]
    d4_gt = y_true_geo[:, :, :, 3]
    theta_gt = y_true_geo[:, :, :, 4]

    d1_pred = y_pred_geo[:, 0, :, :]
    d2_pred = y_pred_geo[:, 1, :, :]
    d3_pred = y_pred_geo[:, 2, :, :]
    d4_pred = y_pred_geo[:, 3, :, :]
    theta_pred = y_pred_geo[:, 4, :, :]

    # d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = torch.split(y_true_geo, split_size_or_sections=5, dim=1)
    # d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = torch.split(y_pred_geo, split_size_or_sections=5, dim=1)
    area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
    area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
    w_union = torch.min(d2_gt, d2_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d3_gt, d3_pred)
    area_intersect = w_union * h_union
    area_union = area_gt + area_pred - area_intersect
    L_AABB = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    L_theta = 1 - torch.cos(theta_pred - theta_gt)
    L_g = L_AABB + 20 * L_theta

    return torch.mean(L_g * y_true_cls * training_mask) + classification_loss
