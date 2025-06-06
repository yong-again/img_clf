import torch


def accuracy(output, target):
    pred = torch.argmax(output, dim=1)  # 가장 높은 logit 값을 가진 클래스 인덱스
    correct = torch.sum(pred == target).item()
    return correct / target.size(0)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
