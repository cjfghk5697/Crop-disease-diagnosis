# augementation 모음집(made by 한철화)
import numpy as np
import matplotlib.pyplot as plt
import torch

""" 사용 예시 BETA=1.0
      if BETA > 0 and np.random.random()>0.5: # cutmix 작동될 확률      
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(img.size()[0]).to(device)
        target_a = label
        target_b = label[rand_index]            
        bbx1, bby1, bbx2, bby2 = utils.rand_bbox(img.size(), lam)
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
        output = model(img,csv_feature)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
"""
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
  
"""
사용 예시
    for batch in (train_loader):
        optimizer.zero_grad()
        x = torch.tensor(batch[0], dtype=torch.float32, device=device)
        y = torch.tensor(batch[1], dtype=torch.long, device=device)

        x, targets_a, targets_b, lam = mixup_data(x, y)
        x, targets_a, targets_b = map(Variable, (x, targets_a, targets_b))

        #outputs = model(x)

        with torch.cuda.amp.autocast():
            pred = model(x)
        loss = mixup_criterion(criterion, pred, targets_a, targets_b, lam)

"""
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def show_plot(x,y,xlabel,ylabel,title,plot_label_1,plot_label_2):
  plt.grid()
  plt.plot(x, label=plot_label_1)
  plt.plot(y, label=plot_label_2)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.title(title, fontsize=25)
  plt.legend()
  plt.show()