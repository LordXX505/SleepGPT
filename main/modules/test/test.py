import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from main.modules.long_net import DilatedAttention  # 替换为你的文件名和类名

# 假设你的 DilatedAttention 类已定义好
class DilatedAttentionTest:
    def __init__(self):
        self.DilatedAttention = DilatedAttention
        # 设置测试参数
        batch_size = 2
        seq_len = 1500
        dim = 512
        num_heads = 8
        segment_lengths = [2, 4]  # 不同的分段长度
        dilated_ratios = [1, 2]  # 不同的膨胀率



        # 实例化你的 DilatedAttention 类
        self.dilated_attention = self.DilatedAttention(
            dim=dim,
            num_heads=num_heads,
            segment_lengths=segment_lengths,
            dilated_ratios=dilated_ratios,
            dropout=0.1
        )
    def run_test(self):
        batch_size = 2
        seq_len = 5
        dim = 512
        num_heads = 8
        segment_lengths = [2, 4]  # 不同的分段长度
        dilated_ratios = [1, 2]  # 不同的膨胀率
        # 创建随机输入数据
        query = torch.randn(batch_size, seq_len, dim)
        key = torch.randn(batch_size, seq_len, dim)
        value = torch.randn(batch_size, seq_len, dim)

        # 前向传播
        output = self.dilated_attention(query, key, value)

        # 打印输出的形状
        print("Dilated Attention Output Shape:", output)  # 期望: (batch_size, seq_len, dim)



# 测试代码调用
test_instance = DilatedAttentionTest()
test_instance.run_test()