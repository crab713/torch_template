import torch

def topkAcc(preds:torch.Tensor, labels:torch.Tensor, topk=(1,)):
        assert preds.shape[0] == labels.shape[0]
        batch_size = preds.shape[0]
        result = []
        for k in topk:
            cnt = 0
            values, indexs = preds.topk(k)
            for i in range(batch_size):
                if labels[i] in indexs[i]:
                    cnt += 1
            result.append(cnt/batch_size)
        return result