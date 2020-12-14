
"""
Using the groundtruth as the user/system feedback
"""
def reward_fn_feedback(predict, gt):
    if predict == gt:
        return 1.0
    else:
        return 0


def MSE(softmax_res_arr, gt_arr):
    n = len(gt_arr)
    mse = 0.0
    for softmax_res, gt in zip(softmax_res_arr, gt_arr):
        mse += (1.0 - softmax_res[gt]) ** 2
    mse /= n
    return mse

def mse2weight(mse):
    return 1.0 / (mse + 1e-10)