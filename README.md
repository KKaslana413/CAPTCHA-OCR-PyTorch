# CRNN Captcha OCR —— 高鲁棒性验证码识别框架

> 一个面向验证码识别（OCR）的端到端训练框架：**CRNN + CTC**，配合**双分支深度监督**、**温度缩放**、**SpecAugment** 与 **SE 注意力**，在嘈杂与形变场景下也能稳定收敛与解码。

---

## ✨ 为什么选择本项目？（优势概览）

- **字符覆盖广**：内置 62 类字符表（0–9 / A–Z / a–z）+ CTC 空白，共 `len(CHAR_LIST)+1` 个类别，适配主流验证码集合。 
- **更强的特征建模**：CNN+GRU 的 CRNN 主干，卷积侧引入 **SE 注意力** 强化通道选择，时序侧 **三层双向 GRU + 残差投影**，提升长序列稳定性。
- **双头深度监督**：主/辅两个分类头共同训练（CTC），主损失 + 0.3×辅损失，并加入**熵正则**抑制过于尖锐的分布，提升泛化。
- **温度缩放（可学习）**：对 logits 做可学习温度缩放，训练中自动约束置信度，减轻过拟合。
- **稳健的数据增强**：图像侧 **Resize+亮度对比度+轻微仿射**、归一化；序列侧引入 **SpecAugment**（时间/“频率”遮挡）。
- **训练更安全**：CTC 训练配合 **梯度裁剪**、**AdamW**、**ReduceLROnPlateau**、可复现实验的多源随机种子设置。
- **更实用的评测**：评估阶段同时输出 **样本级完全匹配准确率** 与 **字符错误率（CER）**。
- **更聪明的解码**：贪心解码内置 **置信度阈值** 与重复折叠机制，低置信字符会被抑制。

---

## 🧩 项目能做什么（作用）？

- 训练并导出一个对多字符验证码的**端到端识别模型**（从彩色图到字符串）。  
- 在**透视/旋转/光照变化**、**字符间距不一**、**干扰线较多**等场景下，仍能较稳地收敛与输出。  
- 提供**可复现实验配置**与**推理组件**（加载最佳权重并解码）。

---

## 🏗️ 模型架构一览

- **输入尺寸**：默认 `384×96×3` 彩色图。
- **卷积骨干**：3 个 `Conv-BN-ReLU + SE + MaxPool` 级联；输出按宽度展开为时序。 
- **时序编码**：`BiGRU × 3`，带 dropout 与 **残差投影**。 
- **分类头**：主/辅两路线性分类器（与 CTC 搭配）。
- **温度缩放**：可学习参数并在前向中进行裁剪与缩放。

---

## 📚 数据与标注约定

- 数据目录：`train-data/`、`valid-data/`、`test-data/`（默认路径可在 `config.py` 修改）。 
- 文件命名：默认从**文件名的第一个下划线之后**截取作为标签，例如 `img_xYz9.png` → 标签 `xYz9`。只读取 `.png`。
- 自动旋正：若读入图像 `w<h`，会**顺时针旋转 90°** 以保障横向时序展开稳定。
- 训练增强：`Resize → Brightness/Contrast → Affine → Normalize → ToTensor`。

> 训练与验证加载器均使用自定义 `collate_fn`：将图像拼成张量，**标签保留变长 list** 以配合 CTC。

---

## 🚀 快速开始

### 1) 安装依赖
```bash
pip install -r requirements.txt
# 参考：torch, albumentations, opencv-python, numpy, tqdm 等
```

### 2) 准备数据
```
your_project/
  ├─ train-data/   # 训练集（文件名含下划线后的标签）
  ├─ valid-data/   # 验证集
  └─ test-data/    # 可选：测试集
```

### 3) 配置参数
在 `config.py` 中调整输入尺寸、字符表、批大小、学习率、早停轮数等：  
例如 `Bach_Size=32`, `Epochs=150`, `Learning_Rate=3e-4`, `Early_Stop_Patience=10` 等。

### 4) 训练
```bash
python train.py
```
- 使用 **AdamW** 优化器与 **ReduceLROnPlateau** 学习率调度；自动保存 `latest_crnn.pth` 与 `best_crnn.pth`。
- 每轮验证后输出 **Val Acc**，并按验证精度触发调度与“早停计数”。
- 训练中进行 **梯度裁剪（max_norm=5.0）** 提升稳定性。

### 5) 评估与指标
- 评估时输出：**完全匹配准确率** 与 **CER（字符错误率）**。
- 控制台还会打印部分样例的 `Pred | True` 对照，便于快速质检。

---

## 🔍 推理与解码（示例）

- 训练脚本内置 `load_model_from_checkpoint(path, device)` 可直接加载权重用于评估/推理。
- 解码使用**贪心策略**，会**折叠重复**并过滤**低置信字符**（阈值在 `config.py` 中配置）。

> 温馨提示：若需要 Beam Search / LM 融合，可在 `engine.py` 基础上扩展。

---

## ⚙️ 关键训练细节（设计初衷）

- **CTC 配置**：`blank=0, zero_infinity=True`，适配变长标签并避免无效梯度。
- **深度监督**：主/辅两头同时做 CTC，辅头权重 0.3；并加入**熵正则**（权重可在 `config.py` 中调节）。
- **SpecAugment**：对时序特征做时间/“频带”遮挡，缓解过拟合并增强鲁棒性。
- **可复现实验**：统一设置 PyTorch/CUDA/NumPy/python 随机种子，关闭 `cudnn.benchmark`。

---

## 🗂️ 代码结构
```
.
├── config.py        # 全局超参、字符表、数据路径
├── dataset.py       # CaptchaDataset：命名规则解析、图像增强与张量化
├── model.py         # CRNN（CNN+BiGRU）+ SE + 双头分类 + 温度缩放
├── engine.py        # 训练/评估循环、CTC、贪心解码、CER 计算
├── train.py         # 训练入口：DataLoader/优化器/调度器/保存-加载
└── README.md
```

---

## ✅ TODO / 路线图
- [ ] 推理脚本与批量图片评测 CLI
- [ ] Beam Search 解码与 n-gram 语言模型融合
- [ ] 半监督伪标签与合成数据自训练
- [ ] 混合精度训练与更轻量骨干（MobileNet/RepVGG）
- [ ] 部署样例（TorchScript / ONNX / TensorRT）

---

## 📄 许可证
建议使用 MIT 或 Apache-2.0，根据你的业务场景选择。
