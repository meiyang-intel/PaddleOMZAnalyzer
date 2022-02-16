import pytest
import numpy as np


def AffineGrid(theta, grid_shape):
    n = grid_shape[0]
    h = grid_shape[1]
    w = grid_shape[2]
    h_idx = np.repeat(
        np.linspace(-1, 1, h)[np.newaxis, :], w, axis=0).T[:, :, np.newaxis]
    w_idx = np.repeat(
        np.linspace(-1, 1, w)[np.newaxis, :], h, axis=0)[:, :, np.newaxis]
    grid = np.concatenate(
        [w_idx, h_idx, np.ones([h, w, 1])], axis=2)  # h * w * 3
    grid = np.repeat(grid[np.newaxis, :], n, axis=0)  # n * h * w *3

    ret = np.zeros([n, h * w, 2])
    theta = theta.transpose([0, 2, 1])
    for i in range(len(theta)):
        ret[i] = np.dot(grid[i].reshape([h * w, 3]), theta[i])

    return ret.reshape([n, h, w, 2]).astype("float64")


def getGridPointValue(data, x, y):
    data_shape = data.shape
    N = data_shape[0]
    C = data_shape[1]
    in_H = data_shape[2]
    in_W = data_shape[3]
    out_H = x.shape[1]
    out_W = x.shape[2]

    #out = np.zeros(data_shape, dtype='float64')
    out = np.zeros([N, C, out_H, out_W], dtype='float64')
    for i in range(N):
        for j in range(out_H):
            for k in range(out_W):
                if y[i, j, k] < 0 or y[i, j, k] > in_H - 1 or x[
                        i, j, k] < 0 or x[i, j, k] > in_W - 1:
                    out[i, :, j, k] = 0
                else:
                    out[i, :, j, k] = data[i, :, y[i, j, k], x[i, j, k]]

    return out


def clip(x, min_n, max_n):
    return np.maximum(np.minimum(x, max_n), min_n)


def unnormalizeAndClip(grid_slice, max_val, align_corners, padding_mode):
    if align_corners:
        grid_slice = 0.5 * ((grid_slice.astype('float64') + 1.0) * max_val)
    else:
        grid_slice = 0.5 * (
            (grid_slice.astype('float64') + 1.0) * (max_val + 1)) - 0.5

    if padding_mode == "border":
        grid_slice = clip(grid_slice, 0, max_val)
    elif padding_mode == "reflection":
        double_range = 2 * max_val if align_corners else (max_val + 1) * 2
        grid_abs = np.abs(grid_slice) if align_corners else np.abs(grid_slice +
                                                                   0.5)
        extra = grid_abs - np.floor(grid_abs / double_range) * double_range
        grid_slice = np.minimum(extra, double_range - extra)
        grid_slice = grid_slice if align_corners else clip(grid_slice - 0.5, 0,
                                                           max_val)
    return grid_slice


def GridSampler(data,
                grid,
                align_corners=True,
                mode="bilinear",
                padding_mode="zeros"):
    dims = data.shape
    N = dims[0]
    in_C = dims[1]
    in_H = dims[2]
    in_W = dims[3]

    out_H = grid.shape[1]
    out_W = grid.shape[2]

    x = grid[:, :, :, 0]
    y = grid[:, :, :, 1]
    y_max = in_H - 1
    x_max = in_W - 1

    x = unnormalizeAndClip(x, x_max, align_corners, padding_mode)
    y = unnormalizeAndClip(y, y_max, align_corners, padding_mode)

    if mode == "bilinear":
        x0 = np.floor(x).astype('int32')
        x1 = x0 + 1
        y0 = np.floor(y).astype('int32')
        y1 = y0 + 1

        wa = np.tile(((x1 - x) * (y1 - y)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wb = np.tile(((x1 - x) * (y - y0)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wc = np.tile(((x - x0) * (y1 - y)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))
        wd = np.tile(((x - x0) * (y - y0)).reshape((N, 1, out_H, out_W)),
                     (1, in_C, 1, 1))

        va = getGridPointValue(data, x0, y0)
        vb = getGridPointValue(data, x0, y1)
        vc = getGridPointValue(data, x1, y0)
        vd = getGridPointValue(data, x1, y1)

        out = (wa * va + wb * vb + wc * vc + wd * vd).astype('float64')
    elif mode == "nearest":
        x = np.round(x).astype('int32')
        y = np.round(y).astype('int32')
        out = getGridPointValue(data, x, y)
    return out


def test_case0():
    x_shape = (2, 3, 8, 8)  # (N, C, HI , WI)
    grid_shape = (2, 7, 9, 2) # (N, HG, WG, 2)
    theta_shape = (2, 2, 3) # (N, 2, 3) ?
    align_corners = True
    padding_mode = "zeros"
    mode = "bilinear"

    x = np.random.randint(0, 255, x_shape).astype('float64')

    theta = np.zeros(theta_shape).astype('float64')
    for i in range(theta_shape[0]):
        for j in range(2):
            for k in range(3):
                theta[i, j, k] = np.random.rand(1)[0]
    print("theta {}\n".format(theta))
    grid = AffineGrid(theta, grid_shape)
    print("grid {} {}\n".format(grid.shape, grid))

    # out_shape (N, C, HG, WG)
    out = GridSampler(x, grid, align_corners, mode, padding_mode)
    print(out.shape)


def maskrcnn_postprocess():
    import numpy as np
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    np.set_printoptions(precision=2)
    def paste_mask(masks, boxes, im_h, im_w):
        """
        Paste the mask prediction to the original image.
        """
        x0, y0, x1, y1 = paddle.split(boxes, 4, axis=1)
        masks = paddle.unsqueeze(masks, [0, 1])  # (1, 1, 14, 14)
        img_y = paddle.arange(0, im_h, dtype='float32') + 0.5
        img_x = paddle.arange(0, im_w, dtype='float32') + 0.5
        img_y = (img_y - y0) / (y1 - y0) * 2 - 1 # (1, 800)
        img_x = (img_x - x0) / (x1 - x0) * 2 - 1 # (1, 600)
        img_x = paddle.unsqueeze(img_x, [1]) # (1, 1, 800)
        img_y = paddle.unsqueeze(img_y, [2]) # (1, 600, 1)
        N = boxes.shape[0]

        gx = paddle.expand(img_x, [N, img_y.shape[1], img_x.shape[2]]) # (1, 600, 800)
        gy = paddle.expand(img_y, [N, img_y.shape[1], img_x.shape[2]]) # (1, 600, 800)
        grid = paddle.stack([gx, gy], axis=3) # (1, 600, 800, 2)
        img_masks = F.grid_sample(masks, grid, align_corners=False) # (1, 1, 600, 800)
        print('paddle.grid_sample result {}'.format(img_masks.numpy()))

        img_masks2 = GridSampler(masks.numpy(), grid.numpy(), align_corners=False, mode='bilinear', padding_mode='zeros') # (1, 1, 600, 800)
        print('refer_impl GridSampler result {}'.format(img_masks2))

        return img_masks[:, 0]

    mask_shape = [2, 14, 14]
    masks = paddle.rand(mask_shape, dtype='float32')

    IMG_H = 100
    IMG_W = 200
    boxes = paddle.to_tensor([[10, 20, 80, 150]], dtype='float32') #x0, y0, x1, y1
    paste_mask(masks[0], boxes, IMG_H, IMG_W)
    
    
if __name__ == '__main__':
     # test_case0()
     maskrcnn_postprocess()