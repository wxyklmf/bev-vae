# BEV-VAE 初学者学习教程

> 本教程基于源码逐步追踪数据流，每一步都标注了 Tensor 的 shape 变化。
> 所有数字均以 nuScenes 默认配置为准：**6 个相机、图像分辨率 256×256、batch_size=1**。

---

## 0. 核心参数速查

| 参数 | 值 | 含义 |
|------|-----|------|
| `V` | 6 | 相机数量（nuScenes 环视 6 路） |
| `image_size` | [256, 256] | 每张图像的高×宽 |
| `scene_size` | [8, 128, 128] | 3D BEV 体素网格 D×H×W |
| `frustum_size` | [60, 32, 32] | 解码用视锥体 D×H×W |
| `pc` | [-51.2,-51.2,-5.0, 51.2,51.2,3.0] | 感知范围：x/y ∈ [-51.2,51.2]m，z ∈ [-5,3]m |
| `embed_dim` | 16 | **BEV 潜空间的通道数**（最终压缩到这里） |

---

## 1. 这个网络解决什么问题？

传统自动驾驶图像生成方法在「图像空间」里做生成，换一套相机就不能用了。

BEV-VAE 的核心思想：**把多视角图像压缩成一张统一的鸟瞰图（BEV）潜在表示**，这个表示与相机无关，所以编码时支持任意相机布局，解码时也能输出任意视角的图像。

```
6张环视图 ──编码──→ BEV 潜空间 ──解码──→ 重建6张图（或任意视角）
(与相机有关)         (与相机无关)          (与相机有关)
```

---

## 2. 整体架构图

```
输入：6张图像，每张 (3, 256, 256)
       ↓
┌──────────────────────────────────────────────────────────────┐
│                         Encoder                              │
│                                                              │
│  Step 1: ImageEncoder  → 图像特征提取 (ViT)                  │
│  Step 2: SceneEncoder  → 图像特征→BEV体素 (3D注意力投影)     │
│  Step 3: StateEncoder  → BEV体素→BEV token序列 (patch压缩)  │
│  Step 4: pre_quant     → 输出 mean 和 logvar                 │
│  Step 5: VAE 采样      → z ~ N(mean, std)                   │
└──────────────────────────────────────────────────────────────┘
       ↓  z: (B, 1024, 16)  ← BEV 潜空间
┌──────────────────────────────────────────────────────────────┐
│                         Decoder                              │
│                                                              │
│  Step 6: post_quant    → 维度恢复                            │
│  Step 7: StateDecoder  → BEV token→BEV体素                  │
│  Step 8: SceneDecoder  → BEV体素→视锥体图像特征 (反投影)     │
│  Step 9: ImageDecoder  → 图像特征→像素 (ViT 解码)            │
└──────────────────────────────────────────────────────────────┘
       ↓
输出：重建的6张图像，每张 (3, 256, 256)
```

---

## 3. 逐步数据流（含 Shape）

> **符号约定**：B=batch_size, V=6(相机数), E=通道/特征维度

---

### 输入数据

数据集提供以下内容：

| 字段 | Shape | 说明 |
|------|-------|------|
| `imgs` | `(B, V, 3, 256, 256)` | 6路相机的 RGB 图像 |
| `intrinsic` | `(B, V, 3, 3)` | 每个相机的内参矩阵（焦距、主点） |
| `extrinsic` | `(B, V, 4, 4)` | 每个相机的外参矩阵（相机→世界坐标变换） |
| `geometric` | `(B, V, 3, 3)` | 图像几何变换矩阵（用于畸变校正等） |

进入编码器前，图像被合并视角维度：

```
(B, V, 3, 256, 256)  →  rearrange  →  (B×V, 3, 256, 256)
    即 (B×6, 3, 256, 256)
```

---

### Step 1：ImageEncoder — 图像特征提取

**目标**：把每张图像切成 patch，用 Transformer 提取特征，得到 2D 特征图。

**实现**：ViT（Vision Transformer）风格，patch_size=8

```
输入:  (B×V, 3, 256, 256)

PatchEmbed (Conv2d, stride=8, 将 8×8 像素块打包成 1 个 token)
  → (B×V, 32, 32, 768)    ← 每张图变成 32×32=1024 个 token，每个维度 768

rearrange: 展平空间维度
  → (B×V, 1024, 768)

RoFormerPE2D Transformer (12层, 带旋转位置编码)
  → (B×V, 1024, 768)      ← shape 不变，但每个 token 已融合了全局上下文

rearrange: 恢复空间维度
  → (B×V, 768, 32, 32)    ← 最终图像特征图

输出:  (B×V, 768, 32, 32)
```

> **直觉理解**：把每张 256×256 图像"分格"成 32×32 的特征地图，每个格子里有 768 维的特征向量描述这块区域的内容。

---

### Step 2：SceneEncoder — 图像特征投影到 BEV 体素

**目标**：把 6 路相机的 2D 特征，通过相机几何关系，"投影"到 3D BEV 体素空间。

这是整个网络最核心的模块，使用**可变形交叉注意力**（Deformable Attention）。

#### 2a. IFPN — 生成多尺度图像特征

```
输入: (B×V, 768, 32, 32)

fpn1 (上采样 ×4): (B×V, 768, 32, 32) → (B×V, 96, 128, 128)
fpn2 (上采样 ×2): (B×V, 768, 32, 32) → (B×V, 96, 64, 64)
fpn3 (原始尺度):  (B×V, 768, 32, 32) → (B×V, 96, 32, 32)

输出: 3 个尺度的特征图列表
```

#### 2b. BEV 体素查询初始化

```
scene_size = [D=8, H=128, W=128]

scene_embedding (可学习): (H×W=16384, 96)
               ↓ reshape + 重复 D 次
query:  (B, D=8, H=128, W=128, E=96)
     ↓ 加 3D 位置编码 + flatten
query:  (B, D×H×W=131072, 96)
```

#### 2c. 3D→2D 投影采样

对 BEV 中每个 3D 点 (x, y, z)，利用相机内外参计算它在每个相机图像上的投影坐标 (u, v)：

```
3D 体素点坐标 (归一化到感知范围)
       ↓ 乘以感知范围 pc
真实 3D 坐标 (x ∈ [-51.2, 51.2]m, y ∈ [-51.2, 51.2]m, z ∈ [-5, 3]m)
       ↓ extrinsic⁻¹ (世界→相机坐标)
相机坐标系
       ↓ intrinsic (相机→图像平面投影)
图像坐标 (u, v) + 有效性 mask（超出图像范围的点标记为无效）
```

#### 2d. 可变形交叉注意力

```
BEV query: (B, 131072, 96)
Value (图像特征): (V, sum_HW, B, 96)  ← 3 个尺度展平后拼接

Cross-Attention: BEV 中每个点去图像特征图上对应位置采样特征
  → (B, 131072, 96)
  ↓ reshape
  → (B, 96, D=8, H=128, W=128)  ← 3D BEV 体素特征

输出:  (B, 96, 8, 128, 128)
```

> **直觉理解**：对 BEV 空间中的每个 3D 体素，去问 6 个相机"你能看到这里吗？看到的话给我你的特征"，把各相机的回答融合成这个体素的特征。

---

### Step 3：StateEncoder — BEV 体素压缩为 Token 序列

**目标**：把高分辨率的 3D BEV 体素，用 patch 压缩成更紧凑的 token 序列。

```
输入: (B, 96, D=8, H=128, W=128)

PatchSceneEmbed (patch_size=4, 对 H×W 做 patch，D 方向保持):
  Conv3d(96→6, 1×1×1)    →  (B, 6, 8, 128, 128)
  rearrange               →  (B×8, 6, 128, 128)
  Conv2d(6→96, 4, s=4)   →  (B×8, 96, 32, 32)     ← H/W 从 128 压缩到 32
  rearrange               →  (B, 8, 32, 32, 96)

rearrange:
  (B, 8, 32, 32, 96) → (B, 32×32=1024, 8×96=768)
  ↑                          ↑ 空间 token     ↑ D 维度展开到通道

RoFormerPE3D Transformer (6层，embed_dim=768)
  → (B, 1024, 768)

输出:  (B, 1024, 768)
```

> **直觉理解**：32×32 = 1024 个 BEV token，每个 token 代表 4×4m 的地面区域，包含高度方向 8 层信息（D=8 展开成 768 维特征的一部分）。

---

### Step 4 & 5：VAE 瓶颈层 — 压缩到潜空间

**目标**：用 VAE（变分自编码器）对 BEV 表示做概率建模，实现可生成的潜空间。

```
输入: (B, 1024, 768)

pre_quant: Linear(768 → 32)        ← 32 = embed_dim × 2
  → (B, 1024, 32)                  ← 前 16 维是 mean，后 16 维是 logvar

DiagonalGaussianDistribution:
  mean:   (B, 1024, 16)
  logvar: (B, 1024, 16)  → std = exp(0.5 × logvar)

VAE 重参数化采样: z = mean + std × ε,  ε ~ N(0, I)
  → z: (B, 1024, 16)

BEV 潜空间的空间化形式: (B, 16, 32, 32)  ← 对应 README 中的 "32×32×16 BEV latent"
```

> **这就是 BEV-VAE 的"压缩包"**：把 6 张图像的所有信息，压缩成一个 32×32×16 的 BEV 特征图。Stage 2 的 DiT 扩散模型就是在这个空间里做生成。

---

### Step 6：post_quant — 从潜空间恢复维度

```
输入: z: (B, 1024, 16)

post_quant: Linear(16 → 768)
  → (B, 1024, 768)
```

---

### Step 7：StateDecoder — BEV Token 恢复为体素

与 StateEncoder 完全对称：

```
输入: (B, 1024, 768)

RoFormerPE3D Transformer (6层)
  → (B, 1024, 768)

rearrange: (B, 1024, 768) → (B, d=8, h=32, w=32, e=96)

DePatchSceneEmbed (patch_size=4，反卷积上采样):
  rearrange     →  (B×8, 96, 32, 32)
  ConvTranspose2d(96→6, 4, s=4)  →  (B×8, 6, 128, 128)   ← H/W 从 32 恢复到 128
  rearrange     →  (B, 6, 8, 128, 128)
  Conv3d(6→96)  →  (B, 96, 8, 128, 128)

输出: (B, 96, 8, 128, 128)
```

---

### Step 8：SceneDecoder — BEV 体素反投影到每个相机

**目标**：对每个相机，从 3D BEV 体素中采样出该相机视角的图像特征（与 SceneEncoder 方向相反）。

#### 8a. SFPN — 生成多尺度 BEV 特征

```
输入: (B, 96, 8, 128, 128)

fpn1 (3D 下采样 ×4): → (B, 96, 2, 32, 32)
fpn2 (3D 下采样 ×2): → (B, 96, 4, 64, 64)
fpn3 (原始尺度):      → (B, 96, 8, 128, 128)

输出: 3 个尺度的 3D 特征图列表
```

#### 8b. 视锥体查询初始化

```
frustum_size = [D=60, H=32, W=32]
（图像空间的视锥体，D=60 代表深度采样层数）

frustum_embedding (可学习): (H×W=1024, 96)
query: (B, D=60, H=32, W=32, 96)
     ↓ flatten + 重复 V 次
query: (B, V=6, D×H×W=61440, 96)
```

#### 8c. 3D 坐标变换（编码器的逆过程）

```
视锥体 3D 坐标 (u×z, v×z, z)
       ↓ geometric⁻¹ 和 intrinsic⁻¹
相机坐标系
       ↓ extrinsic（相机→世界）
3D 世界坐标，归一化到 BEV 范围 [0,1]
```

#### 8d. 交叉注意力采样

```
视锥体 query 从 BEV 3D 特征中采样
  → (B×V, H×W=1024, 96)
  ↓ proj: Linear(96→768)
  → (B×V, 1024, 768)
  ↓ rearrange
  → (B×V, 768, 32, 32)  ← 每个相机的图像特征（32×32 分辨率）

输出:  (B×V, 768, 32, 32)
```

---

### Step 9：ImageDecoder — 图像特征还原为像素

与 ImageEncoder 完全对称：

```
输入: (B×V, 768, 32, 32)

rearrange: → (B×V, 1024, 768)

RoFormerPE2D Transformer (12层)
  → (B×V, 1024, 768)

rearrange: → (B×V, 32, 32, 768)

DePatchEmbed (ConvTranspose2d, stride=8，将 1 个 token 扩展为 8×8 像素):
  → (B×V, 3, 256, 256)   ← 重建出来的 RGB 图像

输出:  (B×V, 3, 256, 256)
     ↓ rearrange
     (B, V=6, 3, 256, 256)  ← 最终重建的 6 张图像
```

---

## 4. 全流程 Shape 总览

```
原始输入                      (B, 6, 3, 256, 256)
        ↓  合并视角
图像输入                      (B×6, 3, 256, 256)
        ↓  Step 1: ImageEncoder (PatchEmbed + ViT)
图像特征图                    (B×6, 768, 32, 32)
        ↓  Step 2: SceneEncoder (IFPN + 可变形注意力)
3D BEV 体素                   (B, 96, 8, 128, 128)
        ↓  Step 3: StateEncoder (PatchEmbed + Transformer)
BEV token                     (B, 1024, 768)
        ↓  Step 4: pre_quant (Linear)
VAE moments                   (B, 1024, 32)  → mean: (B,1024,16) + logvar: (B,1024,16)
        ↓  Step 5: 重参数化采样
BEV 潜变量 z                  (B, 1024, 16)  =  (B, 16, 32, 32) 空间形式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        ↓  Step 6: post_quant (Linear)
解码输入                      (B, 1024, 768)
        ↓  Step 7: StateDecoder
BEV 体素                      (B, 96, 8, 128, 128)
        ↓  Step 8: SceneDecoder (SFPN + 可变形注意力反投影)
每个相机图像特征               (B×6, 768, 32, 32)
        ↓  Step 9: ImageDecoder (ViT + DePatchEmbed)
重建图像                      (B×6, 3, 256, 256)
        ↓  rearrange
最终输出                      (B, 6, 3, 256, 256)
```

---

## 5. 训练目标（Loss）

```
总损失 = 重建损失 + KL 损失 + 感知损失 + 对抗损失

重建损失: L1(重建图像, 原始图像)
KL 损失:  KL(后验分布 q(z|x) ‖ 标准正态 N(0,I))  ← VAE 正则化
感知损失: LPIPS(重建图像, 原始图像)  ← 让高层语义更接近，而不只是像素
对抗损失: PatchGAN 判别器  ← 提升重建图像的真实感
```

---

## 6. 与 Stage 2（DiT）的关系

```
Stage 1 (本仓库):
  真实图像  →  Encoder  →  z (32×32×16)  →  Decoder  →  重建图像

Stage 2 (DiT扩散模型，训练代码未开源):
  噪声 + 3D目标框条件  →  DiT  →  z (32×32×16)  →  Decoder  →  生成图像
                                      ↑
                              Stage 1 的 Decoder 直接复用
```

Stage 2 训练完后，只需要给一个 3D 目标框，DiT 就能生成对应的 BEV 潜变量，再用 Stage 1 的 Decoder 解码出任意视角的图像。

---

## 7. 关键技术索引

| 技术 | 所在文件 | 作用 |
|------|---------|------|
| RoPE 旋转位置编码 | `models/rope.py` | 给 Transformer 注入 2D/3D 位置信息，比绝对位置编码更灵活 |
| 可变形注意力 (MSDA) | `models/layers.py` | 不对所有位置做注意力，只在预测的关键点附近采样，效率更高 |
| FPN 多尺度特征 | `models/stage1/fpn.py` | 同时利用细节特征（高分辨率）和语义特征（低分辨率） |
| DiagonalGaussianDistribution | `models/distributions.py` | VAE 的概率分布实现，支持 KL 散度计算和重参数化采样 |
| PatchSceneEmbed | `models/layers.py` | 对 3D 特征图做 patch 化压缩，类比 ViT 的 PatchEmbed |

---

## 8. 推荐学习顺序

1. **先理解 VAE 基础**：为什么要有 mean/logvar？重参数化采样是什么？
2. **看 `distributions.py`**：BEV-VAE 的概率建模基础
3. **看 `image_layers.py`**：最简单的编解码器，就是 ViT
4. **看 `fpn.py`**：多尺度特征提取，理解为什么需要不同分辨率
5. **看 `scene_layers.py`**：最核心最复杂，重点理解 `point_sampling` 函数（如何用相机参数把 3D 点投影到 2D 图像）
6. **看 `state_layers.py`**：理解 3D patch 压缩
7. **看 `bev_vae.py`**：把前面所有模块串起来，特别是 `log_images` 函数，它展示了完整的中间特征可视化
