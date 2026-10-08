import torch

# 看 transpose 到底改了什么
y = torch.arange(24).reshape(2, 3, 4)

print("y",y)
print(y.shape)
print('y00: ',y[0, 0])                    # 4x3 矩阵
print(y.transpose(-2,-1)[0, 0])   # 3x4 矩阵，是上面那个的转置


x = torch.arange(2*3*4).reshape(2, 3, 4)

# 只看第 0 页
print(x[0])                    # (3, 4)

# 转置后，还是只看第 0 页
y = x.transpose(-2, -1)
print(y[0])                    # (4, 3)，是 x[0] 的转置

# 验证：y[0] 是否等于 x[0].T
print(torch.equal(y[0], x[0].T))   # True