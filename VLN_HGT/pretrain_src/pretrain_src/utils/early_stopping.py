class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        """
        Args:
            patience (int): 没有改进的步骤数，在这之后训练将会停止。默认值为7。
            min_delta (float): 为了被认为是一次改进，监控的数量必须超过这个最小变化量。默认值为0。
        """
        self.patience = patience  # 设定的耐心值，即在多少个验证周期内，如果没有见到损失改善，则停止训练。
        self.min_delta = min_delta  # 设定的最小变化量，验证损失需要改善超过这个值才算是真正的改善。
        self.counter = 0  # 计数器，用来计算没有改进的连续周期数。
        self.best_score = None  # 存储最佳分数（即最低的验证损失）。
        self.early_stop = False  # 早停标志，如果为True，则训练过程需要停止。

    def __call__(self, val_loss):
        score = -val_loss  # 将验证损失转化为分数，因为类是假设分数越高越好。

        if self.best_score is None:
            self.best_score = score  # 如果是第一次调用，初始化最佳分数。
        elif score < self.best_score + self.min_delta:
            self.counter += 1  # 如果当前分数不比之前的最佳分数高（考虑到min_delta），增加计数器。
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # 如果计数器达到耐心阈值，设置早停标志为True。
        else:
            self.best_score = score  # 如果当前分数是新的最佳分数，重置最佳分数和计数器。
            self.counter = 0

