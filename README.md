# Ab Affinity Transformer

## 项目简介
本项目开发了基于Transformer深度学习模型的抗体-抗原亲和力预测系统。
通过处理AB1101数据集中的氨基酸序列，生成PSSM特征矩阵，训练模型预测突变引起的结合自由能变化（ΔΔG），为抗体优化提供AI辅助工具。

> 请注意！本项目不提供具体训练数据，仅提供训练模型脚本以供参考使用

## 设计框架
- 使用PSI-BLAST工具将抗体/抗原氨基酸序列转化为PSSM特征矩阵，捕获序列的进化保守性和位点特异性信息。
- 采用Transformer的Encoder-Decoder结构。Encoder处理序列特征，Decoder预测亲和力变化值。
- 使用PyTorch框架训练，采用MSE损失函数，配合Early Stopping防止过拟合。经过超参数调优，模型可在17个Epoch内验证集损失降低。
- 使用PCC、MAE、R²等多指标评估训练结果。