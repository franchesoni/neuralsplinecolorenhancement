"""Pytorch routines for color conversions and management.

All color arguments are given as 4-dimensional tensors representing batch 
of images (Bx3xHxW).
sRGB values are supposed in the range 0-1 (but values outside the range are
tolerated).

Some examples:

>>> rgb = torch.tensor([0.8, 0.4, 0.2]).view(1, 3, 1, 1)
>>> lab = rgb2lab(rgb)
>>> print(lab.view(-1))
tensor([ 54.6400,  36.9148,  46.1227])

>>> rgb2 = lab2rgb(lab)
>>> print(rgb2.view(-1))
tensor([ 0.8000,  0.4000,  0.2000])

>>> rgb3 = torch.tensor([0.1333,0.0549,0.0392]).view(1, 3, 1, 1)
>>> lab3 = rgb2lab(rgb3)
>>> print(lab3.view(-1))
tensor([ 6.1062,  9.3593,  5.2129])

"""

import torch


# Helper for the creation of module-global constant tensors
def _t(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(
        data, requires_grad=False, dtype=torch.float32, device=device
    )


# Helper for color matrix multiplication
def _mul(coeffs, image):
    # # This is implementation is clearly suboptimal.  The function will
    # # be implemented with 'einsum' when a bug in pytorch 0.4.0 will be
    # # fixed (Einsum modifies variables in-place #7763).
    # coeffs = coeffs.to(image.device)
    # r0 = image[:, 0:1, :, :].repeat(1, 3, 1, 1) * coeffs[:, 0].view(1, 3,
    #  1, 1)
    # r1 = image[:, 1:2, :, :].repeat(1, 3, 1, 1) * coeffs[:, 1].view(1, 3,
    #  1, 1)
    # r2 = image[:, 2:3, :, :].repeat(1, 3, 1, 1) * coeffs[:, 2].view(1, 3,
    #  1, 1)
    # return r0 + r1 + r2
    return torch.einsum("dc,bcij->bdij", (coeffs.to(image), image))


_RGB_TO_XYZ = {
    "srgb": _t(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    ),
    "prophoto": _t(
        [
            [0.7976749, 0.1351917, 0.0313534],
            [0.2880402, 0.7118741, 0.0000857],
            [0.0000000, 0.0000000, 0.8252100],
        ]
    ),
}


WHITE_POINTS = {
    item[0]: _t(item[1:]).view(1, 3, 1, 1)
    for item in [
        ("a", 1.0985, 1.0000, 0.3558),
        ("b", 0.9807, 1.0000, 1.1822),
        ("e", 1.0000, 1.0000, 1.0000),
        ("d50", 0.9642, 1.0000, 0.8251),
        ("d55", 0.9568, 1.0000, 0.9214),
        ("d65", 0.9504, 1.0000, 1.0888),
        ("icc", 0.9642, 1.0000, 0.8249),
    ]
}


_EPSILON = 0.008856
_KAPPA = 903.3
_XYZ_TO_LAB = _t(
    [[0.0, 116.0, 0.0], [500.0, -500.0, 0.0], [0.0, 200.0, -200.0]]
)
_LAB_OFF = _t([16.0, 0.0, 0.0]).view(1, 3, 1, 1)


def remove_gamma(rgb, gamma="srgb"):
    """Gamma to linear rgb.

    Assume that rgb values are in the [0, 1] range (but values outside are 
    tolerated).

    gamma can be "srgb", a real-valued exponent, or None.

    >>> remove_gamma(apply_gamma(torch.tensor([0.001, 0.3, 0.4])))
    tensor([ 0.0010,  0.3000,  0.4000])

    >>> remove_gamma(torch.tensor([0.5, 0.4, 0.1]).view([1, 3, 1, 1]), 2.0).view(-1)
    tensor([ 0.2500, 0.1600, 0.0100])
    """
    if gamma == "srgb":
        T = 0.04045
        rgb1 = torch.max(rgb, rgb.new_tensor(T))
        return torch.where(
            rgb < T,
            rgb / 12.92,
            torch.pow(torch.abs(rgb1 + 0.055) / 1.055, 2.4),
        )
    elif gamma is None:
        return rgb
    else:
        res = torch.pow(
            torch.max(rgb, rgb.new_tensor(0.0)), float(gamma)
        ) + torch.min(
            rgb, rgb.new_tensor(0.0)
        )  # very important to avoid vanishing gradients
        return res


def rgb2xyz(rgb, gamma_correction="srgb", clip_rgb=False, space="srgb"):
    """sRGB to XYZ conversion.

    rgb:  Bx3xHxW
    return: Bx3xHxW

    >>> rgb2xyz(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([ 0.,  0.,  0.])

    >>> rgb2xyz(torch.tensor([0., 0.75, 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([ 0.1868,  0.3737,  0.0623])

    >>> rgb2xyz(torch.tensor([0.4, 0.8, 0.2]).view(1, 3, 1, 1), gamma_correction=None).view(-1)
    tensor([ 0.4871,  0.6716,  0.2931])

    >>> rgb2xyz(torch.ones(2, 3, 4, 5)).size()
    torch.Size([2, 3, 4, 5])

    >>> xyz2rgb(torch.tensor([-1, 2., 0.]).view(1, 3, 1, 1), clip_rgb=True).view(-1)
    tensor([ 0.0000,  1.0000,  0.0000])

    >>> rgb2xyz(torch.tensor([0.4, 0.8, 0.2]).view(1, 3, 1, 1), gamma_correction=None, space='prophoto').view(-1)
    tensor([ 0.4335,  0.6847,  0.1650])

    """
    if clip_rgb:
        rgb = torch.clamp(rgb, 0, 1)
    rgb = remove_gamma(rgb, gamma_correction)
    return _mul(_RGB_TO_XYZ[space], rgb)


def _lab_f(x):
    x1 = torch.max(x, x.new_tensor(_EPSILON))
    return torch.where(
        x > _EPSILON, torch.pow(x1, 1.0 / 3), (_KAPPA * x + 16.0) / 116.0
    )


def xyz2lab(xyz, white_point="d65"):
    """XYZ to Lab conversion.

    xyz: Bx3xHxW
    return: Bx3xHxW

    >>> xyz2lab(torch.tensor([0., 0., 0.]).view(1, 3, 1, 1)).view(-1)
    tensor([ 0.,  0.,  0.])

    >>> xyz2lab(torch.tensor([0.4, 0.2, 0.1]).view(1, 3, 1, 1)).view(-1)
    tensor([ 51.8372,  82.3018,  26.7245])

    >>> xyz2lab(torch.tensor([1., 1., 1.]).view(1, 3, 1, 1), white_point="e").view(-1)
    tensor([ 100., 0., 0.])

    """
    xyz = xyz / WHITE_POINTS[white_point].to(xyz)
    f_xyz = _lab_f(xyz)
    return _mul(_XYZ_TO_LAB, f_xyz) - _LAB_OFF.to(xyz)


def rgb2lab(
    rgb,
    white_point="d65",
    gamma_correction="srgb",
    clip_rgb=False,
    space="srgb",
):
    """sRGB to Lab conversion."""
    lab = xyz2lab(rgb2xyz(rgb, gamma_correction, clip_rgb, space), white_point)
    return lab


def squared_deltaE94(lab1, lab2):
    """Squared Delta E (CIE 1994).

    Default parameters for the 'Graphic Art' version.

    lab1: Bx3xHxW   (reference color)
    lab2: Bx3xHxW   (other color)
    return: Bx1xHxW

    """
    diff_2 = (lab1 - lab2) ** 2
    dl_2 = diff_2[:, 0:1, :, :]
    c1 = torch.norm(lab1[:, 1:3, :, :], 2, 1, keepdim=True)
    c2 = torch.norm(lab2[:, 1:3, :, :], 2, 1, keepdim=True)
    dc_2 = (c1 - c2) ** 2
    dab_2 = torch.sum(diff_2[:, 1:3, :, :], 1, keepdim=True)
    dh_2 = torch.abs(dab_2 - dc_2)
    de_2 = (
        dl_2 + dc_2 / ((1 + 0.045 * c1) ** 2) + dh_2 / ((1 + 0.015 * c1) ** 2)
    )
    return de_2
