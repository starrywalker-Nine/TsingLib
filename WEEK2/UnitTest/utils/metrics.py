def mse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("预测和目标的长度必须相等")
    return sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)

def mae(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("预测和目标的长度必须相等")
    return sum(abs(p - t) for p, t in zip(predictions, targets)) / len(predictions)

def rmse(predictions, targets):
    if len(predictions) != len(targets):
        raise ValueError("预测和目标的长度必须相等")
    return (sum((p - t) ** 2 for p, t in zip(predictions, targets)) / len(predictions)) ** 0.5
