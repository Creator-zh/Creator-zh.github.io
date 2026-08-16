---
layout: post
title: "从 DeltaNet 到 Non-Diagonal Preconditioning：我们为什么开始关心 Key 的相关性"
date: 2026-08-16 12:00:00
permalink: /blog/2026/deltanet-nondiagonal-preconditioning/
description: "一份关于 DeltaNet、预条件化、Key 相关性与低秩子空间 U 的阶段性研究笔记。"
tags: [DeltaNet, preconditioning, research]
categories: [research-notes]
featured: false
giscus_comments: false
---

# 从 DeltaNet 到 Non-Diagonal Preconditioning：我们为什么开始关心 Key 的相关性

> 这是一份阶段性研究笔记。目标不是直接给出最终方法，而是把问题推到一个足够清楚的位置：  
> **DeltaNet 为什么需要 preconditioning；Preconditioned DeltaNet 为什么采用 diagonal；diagonal 丢掉了什么；我们为什么自然地开始研究 correlation，以及其中可能存在的低秩子空间 $U$。**

> Typora：偏好设置 → Markdown → 勾选 **Inline Math**。行内用 `$...$`，独立公式用单独一行的 `$$...$$`。

---

## 0. 先说问题

DeltaNet 的更新很简单：

$$
\mathbf{S}_t
=
\mathbf{S}_{t-1}
+
\beta_t
\left(
\mathbf{v}_t-\mathbf{S}_{t-1}\mathbf{k}_t
\right)
\mathbf{k}_t^\top .
$$

这里：

- $\mathbf{S}_t\in\mathbb{R}^{d_v\times d_k}$：fast-weight memory；
- $\mathbf{k}_t\in\mathbb{R}^{d_k}$：key，也是当前样本在参数更新中的输入方向；
- $\mathbf{v}_t\in\mathbb{R}^{d_v}$：希望写入的 value；
- $\mathbf{q}_t\in\mathbb{R}^{d_k}$：query，输出为 $\mathbf{o}_t=\mathbf{S}_t\mathbf{q}_t$；
- $\beta_t\in[0,1]$：DeltaNet 主状态更新的写入强度。

这个式子的问题不在于它是不是一个合理的 delta rule，而在于：

$$
\boxed{\text{不同 key 方向是否应该使用相同的更新尺度？}}
$$

如果 key 的各个维度独立、尺度接近、主方向恰好和坐标轴对齐，那么问题不大。

但如果 key 存在明显的二阶结构：

$$
\mathbb{E}[\mathbf{k}\mathbf{k}^\top]
\neq
\operatorname{Diag}\bigl(\mathbb{E}[\mathbf{k}\odot\mathbf{k}]\bigr),
$$

那么仅仅沿坐标轴调整每一维的大小，可能不够。

我们现在研究的就是这个缺口。

---

# 1. DeltaNet：把 memory update 看成在线回归

对当前 token 定义平方损失

$$
\mathcal{L}_t(\mathbf{S})
=
\frac12
\left\|
\mathbf{S}\mathbf{k}_t-\mathbf{v}_t
\right\|_2^2.
$$

梯度为

$$
\nabla_{\mathbf{S}}\mathcal{L}_t
=
(\mathbf{S}\mathbf{k}_t-\mathbf{v}_t)\mathbf{k}_t^\top.
$$

做一步 SGD：

$$
\mathbf{S}_t
=
\mathbf{S}_{t-1}
-
\beta_t\nabla_{\mathbf{S}}\mathcal{L}_t(\mathbf{S}_{t-1}),
$$

就得到

$$
\boxed{
\mathbf{S}_t
=
\mathbf{S}_{t-1}
+
\beta_t
(\mathbf{v}_t-\mathbf{S}_{t-1}\mathbf{k}_t)
\mathbf{k}_t^\top .
}
$$

所以 DeltaNet 可以理解为：  
**对一个在线线性回归问题，每个 token 做一次一阶更新。**

这一步给了我们后面的切入点：既然是优化问题，就可以问它的 curvature 是什么。

---

# 2. DeltaNet 忽略了什么：Key Gram

考虑累计最小二乘

$$
\min_{\mathbf{S}}
\frac12\sum_{i=1}^{t}
\|\mathbf{S}\mathbf{k}_i-\mathbf{v}_i\|_2^2
+
\frac{\lambda}{2}\|\mathbf{S}\|_F^2.
$$

定义

$$
\mathbf{G}_t
=
\sum_{i=1}^{t}\mathbf{k}_i\mathbf{k}_i^\top
+
\lambda\mathbf{I} ,
$$

以及

$$
\mathbf{C}_t
=
\sum_{i=1}^{t}\mathbf{v}_i\mathbf{k}_i^\top .
$$

则精确最小二乘解满足

$$
\mathbf{S}_t^\star
=
\mathbf{C}_t\mathbf{G}_t^{-1}.
$$

这里的核心对象是

$$
\boxed{
\mathbf{G}_t
=
\sum_i\mathbf{k}_i\mathbf{k}_i^\top+\lambda\mathbf{I}
}
$$

即 key Gram / 二阶矩矩阵。

从优化角度看，它描述不同 key 方向的曲率：

- 高频、大方差方向对应较大的 curvature；
- 低频、小方差方向对应较小的 curvature；
- 非零的 $G_{ij}$ 描述第 $i,j$ 个 key 维度之间的相关性。

DeltaNet 的普通更新并没有显式使用这些信息。

可以粗略理解为：

$$
\text{DeltaNet}
\quad\approx\quad
\text{用 }\mathbf{I}\text{ 作为 key-space metric}.
$$

而精确二阶校正会涉及

$$
\mathbf{G}_t^{-1}.
$$

---

# 3. 为什么不能直接使用 full $\mathbf{G}_t^{-1}$

理论上最自然的是

$$
\tilde{\mathbf{k}}_t
=
\mathbf{G}_t^{-1}\mathbf{k}_t.
$$

然后把 write key 从 $\mathbf{k}_t$ 换成 $\tilde{\mathbf{k}}_t$。

问题也很直接。

### 3.1 计算量

$$
\mathbf{G}_t\in\mathbb{R}^{d_k\times d_k}.
$$

显式维护 full matrix 需要

$$
O(d_k^2)
$$

状态和更新开销。

### 3.2 求逆稳定性

如果 $\mathbf{G}_t$ 的谱跨越多个数量级，那么

$$
\mathbf{G}_t^{-1}
$$

会放大小特征值方向。

这也是我们不希望直接把“更接近 full inverse”当成唯一目标的原因。

### 3.3 并行性

精确逆可以写成 Sherman--Morrison 递推：

$$
\mathbf{P}_t
=
\mathbf{P}_{t-1}
-
\frac{
\mathbf{P}_{t-1}\mathbf{k}_t\mathbf{k}_t^\top\mathbf{P}_{t-1}
}{
1+\mathbf{k}_t^\top\mathbf{P}_{t-1}\mathbf{k}_t
}.
$$

其中

$$
\mathbf{P}_t=\mathbf{G}_t^{-1}.
$$

这个更新显式依赖 $\mathbf{P}_{t-1}$，不具有 DeltaNet 所需要的简单 token-parallel / chunk-parallel 结构。

所以真正的问题不是

$$
\text{“能不能计算 full inverse？”}
$$

而是

$$
\boxed{
\text{能否保留有用的二阶结构，同时仍然高效、稳定、可并行？}
}
$$

---

# 4. Preconditioned DeltaNet：先退到 diagonal

Preconditioned DeltaNet 的一个关键选择，是不再维护完整的 $\mathbf{G}_t$，而只保留逐维二阶统计。

记

$$
\mathbf{A}_t\in\mathbb{R}^{d_k},
$$

其递推为

$$
\boxed{
\mathbf{A}_t
=
\alpha_t^P\mathbf{A}_{t-1}
+
\beta_t^P(\mathbf{k}_t\odot\mathbf{k}_t).
}
$$

这里：

- $\alpha_t^P$：preconditioner memory 的衰减；
- $\beta_t^P$：当前 key 对 preconditioner statistics 的写入强度；
- $\mathbf{A}_t$：每个 key 维度自己的累计尺度；
- 上标 $P$ 表示这组参数属于 preconditioner，不等于主 DeltaNet 的 $\alpha_t,\beta_t$。

如果把它写成矩阵，

$$
\mathbf{D}_t=\operatorname{Diag}(\mathbf{A}_t),
$$

那么本质上只保留了 full Gram 的对角部分：

$$
\mathbf{G}_t
\quad\longrightarrow\quad
\mathbf{D}_t.
$$

这一步非常重要，因为 diagonal recurrence 可以逐元素 scan，适合 chunkwise parallelization。

---

# 5. PDN 为什么还需要 bounded mapping

如果直接使用

$$
\mathbf{A}_t^{-1},
$$

仍然会有数值问题。

PDN 观察到原始 $\mathbf{A}_t$ 具有明显的重尾 / 近似 log-normal 行为，因此最终并不是简单做 reciprocal，而是先进入 log-domain：

$$
\mathbf{r}_t
=
\log\mathbf{A}_t-\mu,
$$

再做有界压缩：

$$
\mathbf{s}_t
=
\frac{\mathbf{r}_t}{1+|\mathbf{r}_t|},
$$

最后

$$
\mathbf{B}_t
=
\exp\left(
-\log(x)\mathbf{s}_t
\right).
$$

于是逐元素有

$$
\boxed{
\frac1x
\le B_{t,j}\le x.
}
$$

最终 write key 为

$$
\tilde{\mathbf{k}}_t
=
\mathbf{B}_t\odot\mathbf{k}_t.
$$

状态更新：

$$
\mathbf{S}_t
=
\mathbf{S}_{t-1}
+
\beta_t
(\mathbf{v}_t-\mathbf{S}_{t-1}\mathbf{k}_t)
\tilde{\mathbf{k}}_t^\top.
$$

这里真正值得保留的想法不是某个具体 squash 公式，而是：

$$
\boxed{
\text{preconditioning 不必精确逼近 inverse；它首先要处于一个稳定的范围内。}
}
$$

这对我们后面的 non-diagonal 设计很重要。

---

# 6. Diagonal 到底丢掉了什么

设 full Gram 为

$$
\mathbf{G}_t
=
\begin{bmatrix}
G_{11} & G_{12} & \cdots\\
G_{21} & G_{22} & \cdots\\
\vdots & \vdots & \ddots
\end{bmatrix}.
$$

Diagonal approximation 只保留

$$
G_{11},G_{22},\ldots,G_{d_k d_k},
$$

丢掉

$$
G_{ij},\qquad i\neq j.
$$

这意味着 PDN 可以回答：

> 第 $j$ 个维度整体上是不是过大或过小？

但它不能直接回答：

> 哪几个维度总是一起变化？

也不能表示一个不与坐标轴对齐的主方向。

例如二维情况下，如果 key 主要沿

$$
\frac1{\sqrt2}(1,1)
$$

变化，那么它的主要二阶方向是一个旋转后的方向，而不是 $x$ 轴或 $y$ 轴。

Diagonal scaling 能改变两个坐标轴的长度，却不能表达这个旋转方向。

所以我们从这里开始考虑：

$$
\boxed{\text{non-diagonal preconditioning}}
$$

但这里不能直接跳到 full matrix。  
我们需要先把“尺度”和“相关性”分开。

---

# 7. 一个更自然的分解：Magnitude 与 Correlation

定义

$$
\mathbf{D}_t
=
\operatorname{Diag}(\operatorname{diag}(\mathbf{G}_t)).
$$

只要各维方差为正，就可以把 Gram 写成

$$
\boxed{
\mathbf{G}_t
=
\mathbf{D}_t^{1/2}
\mathbf{R}_t
\mathbf{D}_t^{1/2}.
}
$$

其中

$$
\mathbf{R}_t
=
\mathbf{D}_t^{-1/2}
\mathbf{G}_t
\mathbf{D}_t^{-1/2}.
$$

$\mathbf{R}_t$ 可以看成 correlation-like matrix，并且

$$
\operatorname{diag}(\mathbf{R}_t)=\mathbf{1}.
$$

这个分解给我们一个更清楚的视角：

$$
\boxed{
\mathbf{G}_t
=
\underbrace{\mathbf{D}_t}_{\text{axis-wise magnitude}}
+
\underbrace{\mathbf{R}_t}_{\text{cross-dimensional correlation}}
}
$$

更准确地说，两者通过

$$
\mathbf{G}_t=\mathbf{D}_t^{1/2}\mathbf{R}_t\mathbf{D}_t^{1/2}
$$

组合。

PDN 目前主要能够看到第一部分：  
**每个坐标轴自己的尺度。**

我们想研究的是第二部分：

$$
\boxed{
\mathbf{R}_t
}
$$

是否包含可以低成本利用的结构。

---

# 8. 为什么先研究 correlation，而不是直接 sketch raw Gram

这是我们最近讨论后很重要的一点。

如果直接对

$$
\mathbf{G}_t
$$

做低秩近似

$$
\mathbf{G}_t\approx\mathbf{U}\mathbf{\Lambda}\mathbf{U}^\top,
$$

会遇到两个问题。

第一，$\mathbf{G}_t$ 本身可能有很强的尺度长尾。低秩近似会优先拟合能量最大的方向。

第二，我们最终关心的通常不是

$$
\mathbf{G}_t
$$

本身，而是它对 preconditioning 的作用。  
小特征值方向的近似误差，在 inverse-like mapping 中可能被明显放大。

所以我们暂时不把问题写成

$$
\text{“如何低秩近似 }\mathbf{G}_t\text{？”}
$$

而是先把 diagonal magnitude 拿掉：

$$
\mathbf{z}_t
=
\mathbf{D}_t^{-1/2}\mathbf{k}_t.
$$

然后研究 normalized key 的二阶结构

$$
\mathbf{R}_t
\sim
\mathbb{E}[\mathbf{z}_t\mathbf{z}_t^\top].
$$

也就是说，我们想问：

$$
\boxed{
\text{在去掉逐维 heavy-tail scale 之后，剩余 correlation 是否更简单？}
}
$$

这个问题目前还没有答案，需要实验验证。

---

# 9. 从 diagonal 到 non-diagonal：我们真正想增加的只是 correlation correction

如果完全没有跨维相关，

$$
\mathbf{R}_t\approx\mathbf{I}.
$$

那么 diagonal preconditioner 已经足够。

如果存在相关性，可以写成

$$
\mathbf{R}_t
=
\mathbf{I}+\mathbf{E}_t,
$$

其中

$$
\mathbf{E}_t
=
\mathbf{R}_t-\mathbf{I}
$$

只描述 off-axis correlation。

因此，我们不一定需要学习一个任意的

$$
d_k\times d_k
$$

矩阵。

更自然的问题是：

$$
\boxed{
\mathbf{E}_t\text{ 是否集中在一个低维子空间中？}
}
$$

如果答案是肯定的，可以尝试

$$
\boxed{
\mathbf{E}_t
\approx
\mathbf{U}_t\mathbf{H}_t\mathbf{U}_t^\top.
}
$$

其中

$$
\mathbf{U}_t\in\mathbb{R}^{d_k\times r},
\qquad
r\ll d_k,
$$

而

$$
\mathbf{H}_t\in\mathbb{R}^{r\times r}.
$$

也可以进一步限制

$$
\mathbf{H}_t=\operatorname{Diag}(\boldsymbol{\lambda}_t),
$$

得到

$$
\mathbf{E}_t
\approx
\mathbf{U}_t
\operatorname{Diag}(\boldsymbol{\lambda}_t)
\mathbf{U}_t^\top.
$$

这里：

- $\mathbf{U}_t$：相关性主要存在的 $r$ 个方向；
- $\mathbf{H}_t$ 或 $\boldsymbol{\lambda}_t$：这些方向上的相关强度；
- $r$：correlation rank，决定表达能力和计算成本。

于是我们从

$$
\text{Diagonal}
$$

走向的并不是

$$
\text{Arbitrary Full Matrix},
$$

而是

$$
\boxed{
\text{Diagonal magnitude}
+
\text{Low-rank correlation correction}.
}
$$

---

# 10. 为什么这个 $U$ 不是随便加出来的

这是目前最需要说清楚的一点。

我们并不是先决定“做 low-rank”，再去寻找解释。

逻辑顺序应该是：

$$
\text{DeltaNet}
\rightarrow
\text{需要 curvature information}
$$

$$
\rightarrow
\mathbf{G}_t
$$

$$
\rightarrow
\text{full Gram 太贵且不稳定}
$$

$$
\rightarrow
\text{PDN 保留 diagonal magnitude}
$$

$$
\rightarrow
\text{diagonal 无法表示 cross-dimensional correlation}
$$

$$
\rightarrow
\mathbf{G}_t
=
\mathbf{D}_t^{1/2}\mathbf{R}_t\mathbf{D}_t^{1/2}
$$

$$
\rightarrow
\text{研究 }\mathbf{R}_t-\mathbf{I}
$$

$$
\rightarrow
\boxed{
\text{如果它近似低秩，才引入 }\mathbf{U}.
}
$$

所以 $U$ 是否存在低秩结构，本身就是一个需要验证的研究问题，而不是一个假设后直接成立的结论。

---

# 11. $U$ 应该是什么：目前有三种可能

到这里我们还没有决定最终形式。

## 11.1 固定、可学习的 $U$

每个 head 学一个

$$
\mathbf{U}\in\mathbb{R}^{d_k\times r}.
$$

token 只更新低维统计。

优点：

- 不产生 token-level 的 $U_t\rightarrow U_{t+1}$ 依赖；
- 投影 $\mathbf{k}_t\mapsto\mathbf{U}^\top\mathbf{k}_t$ 可以完全并行；
- 计算量约为 $O(d_k r)$。

问题：

- 一个固定子空间能否覆盖不同序列、不同上下文中的 correlation？
- 不同 layer / head 的低秩性是否一致？

---

## 11.2 Chunk-wise $U_{[i]}$

每个 chunk 根据上一 chunk 或历史统计产生一个低秩子空间。

优点：

- 可以随上下文变化；
- 比 token-wise dynamic $U_t$ 更容易保持 chunk parallelism。

问题：

- 如何构造而不做昂贵 SVD？
- chunk 间更新是否稳定？
- $U_{[i]}$ 的方向跳变会不会破坏训练？

---

## 11.3 Token-wise dynamic $U_t$

理论表达能力最强，但目前最不看好。

因为如果

$$
\mathbf{U}_t
=
F(\mathbf{U}_{t-1},\mathbf{k}_t),
$$

就重新引入了我们想避免的 sequential matrix recurrence。

所以除非存在特殊的 associative / scan form，否则它很可能不适合作为主方案。

---

# 12. 如果 $U$ 存在，我们希望复杂度是什么

Full matrix 需要

$$
O(d_k^2)
$$

级别的存储和乘法。

低秩形式

$$
\mathbf{U}\mathbf{H}_t\mathbf{U}^\top\mathbf{k}_t
$$

可以拆成

$$
\mathbf{h}_t
=
\mathbf{U}^\top\mathbf{k}_t
\in\mathbb{R}^{r},
$$

$$
\mathbf{g}_t
=
\mathbf{H}_t\mathbf{h}_t
\in\mathbb{R}^{r},
$$

$$
\mathbf{U}\mathbf{g}_t
\in\mathbb{R}^{d_k}.
$$

如果 $\mathbf{H}_t$ 为 diagonal，主要开销是

$$
O(d_k r).
$$

如果 $\mathbf{H}_t$ 为 dense $r\times r$，则是

$$
O(d_k r+r^2).
$$

只要

$$
r\ll d_k,
$$

它才有可能比 full non-diagonal matrix 更合理。

---

# 13. 但现在还不能直接把它叫做 preconditioner

即使

$$
\mathbf{R}_t-\mathbf{I}
\approx
\mathbf{U}\mathbf{H}_t\mathbf{U}^\top
$$

成立，我们还缺一层关键设计：

$$
\boxed{
\text{如何把 correlation statistics 映射成稳定的 preconditioning operator？}
}
$$

直接做

$$
\mathbf{R}_t^{-1}
$$

或者

$$
\mathbf{R}_t^{-1/2}
$$

未必是好的答案。

PDN 已经给出一个很重要的提醒：

> 二阶统计本身可以是长尾的；稳定的 attention recurrence 不应该无约束地放大某些方向。

因此 non-diagonal 版本最终也应该满足类似的 boundedness：

$$
m\mathbf{I}
\preceq
\mathbf{P}_t
\preceq
M\mathbf{I},
$$

其中 $0<m<M<\infty$。

但如何从

$$
(\mathbf{D}_t,\mathbf{U}_t,\mathbf{H}_t)
$$

得到这样的 $\mathbf{P}_t$，我们暂时不在这篇笔记中定死。

这是下一步的问题。

---

# 14. 到目前为止，我们真正需要验证的事情

目前最重要的不是继续加公式，而是回答下面几个问题。

### 问题 1：Correlation 是否真的显著？

计算

$$
\mathbf{R}_t
=
\mathbf{D}_t^{-1/2}\mathbf{G}_t\mathbf{D}_t^{-1/2}
$$

后，检查

$$
\|\mathbf{R}_t-\mathbf{I}\|
$$

是否足够大。

如果本身接近单位阵，non-diagonal correction 没有必要。

---

### 问题 2：Correlation residual 是否低秩？

研究

$$
\mathbf{E}_t=\mathbf{R}_t-\mathbf{I}
$$

的特征值 / 奇异值衰减。

重点不是看 raw $\mathbf{G}_t$ 的谱，而是看**去掉 diagonal scale 后**：

$$
\sigma_1(\mathbf{E}_t),
\sigma_2(\mathbf{E}_t),\ldots
$$

是否快速衰减。

这是 $U$ 是否合理的第一性证据。

---

### 问题 3：低秩结构是否稳定？

即使单个 chunk 中低秩，也要看不同时间的 principal subspace 是否稳定。

例如比较

$$
\mathbf{U}_{[i]}^\top\mathbf{U}_{[i+1]}
$$

对应的 principal angles。

如果子空间快速旋转，固定 $U$ 很难成立。

如果变化慢，固定或 chunk-wise $U$ 都可能有意义。

---

### 问题 4：低秩性来自哪里？

需要区分：

$$
\text{raw Gram low-rank}
$$

和

$$
\text{correlation residual low-rank}.
$$

我们现在更关心后者。

因为前者容易被 heavy-tail magnitude 主导，而后者更接近“维度之间真正共享的方向”。

---

### 问题 5：如何不破坏并行性？

最终形式至少应该避免

$$
\mathbf{U}_t
=
F(\mathbf{U}_{t-1},\mathbf{k}_t)
$$

这种不可并行的 token-wise matrix recurrence。

更希望得到：

$$
\boxed{
\text{固定 }U
\quad\text{或}\quad
\text{chunk-wise }U
}
$$

再让低维统计使用 scan / chunkwise recurrence。

---

### 问题 6：如何保证数值稳定？

即使低秩 approximation 很准，也不能直接推出 attention recurrence 稳定。

最终还需要约束 preconditioned write key

$$
\tilde{\mathbf{k}}_t
=
\mathbf{P}_t\mathbf{k}_t
$$

使

$$
\mathbf{I}
-
\beta_t
\mathbf{k}_t
\tilde{\mathbf{k}}_t^\top
$$

保持稳定。

因此我们后续研究的重点不是单纯：

$$
\min\|\mathbf{G}_t-\hat{\mathbf{G}}_t\|,
$$

而更应该是：

$$
\boxed{
\text{在稳定和并行约束下，保留最有用的 non-diagonal geometry。}
}
$$

---

# 15. 当前思路停在哪里

到这里，我们的逻辑可以压缩成四行：

$$
\boxed{
\text{DeltaNet}
:
\text{没有显式 curvature correction}
}
$$

$$
\boxed{
\text{PDN}
:
\text{用逐维二阶统计做 bounded diagonal correction}
}
$$

$$
\boxed{
\text{我们的疑问}
:
\text{key dimensions 存在相关时，diagonal 是否足够？}
}
$$

$$
\boxed{
\text{当前研究点}
:
\mathbf{G}_t
=
\mathbf{D}_t^{1/2}\mathbf{R}_t\mathbf{D}_t^{1/2},
\qquad
\mathbf{R}_t-\mathbf{I}
\stackrel{?}{\approx}
\mathbf{U}\mathbf{H}_t\mathbf{U}^\top .
}
$$

其中问号很重要。

我们现在还没有证明 correlation residual 是低秩的，也还没有决定 $U$ 应该固定、chunk-wise，还是采用其他结构。

下一步应该先用真实训练中的 key statistics 回答这两个问题：

$$
\boxed{
\text{它是不是低秩？}
}
$$

以及

$$
\boxed{
\text{这个低秩子空间是不是稳定到足以被高效利用？}
}
$$

只有这两个问题得到正面答案之后，才值得继续设计具体的 non-diagonal bounded mapping 和并行 kernel。

---

## 附：本文最重要的记号

| 符号 | 含义 |
|---|---|
| $\mathbf{S}_t$ | DeltaNet 的 fast-weight memory |
| $\mathbf{k}_t$ | key / 当前回归样本的输入方向 |
| $\mathbf{v}_t$ | 写入目标 value |
| $\mathbf{q}_t$ | query |
| $\beta_t$ | 主 DeltaNet 更新强度 |
| $\mathbf{G}_t$ | full key Gram / 二阶矩 |
| $\mathbf{A}_t$ | PDN 使用的逐维二阶统计 |
| $\alpha_t^P$ | preconditioner statistics 的衰减 |
| $\beta_t^P$ | preconditioner statistics 的写入强度 |
| $\mathbf{B}_t$ | PDN 的 bounded diagonal scaling |
| $\mathbf{D}_t$ | full Gram 的 diagonal magnitude |
| $\mathbf{R}_t$ | diagonal-normalized correlation-like matrix |
| $\mathbf{E}_t=\mathbf{R}_t-\mathbf{I}$ | correlation residual |
| $\mathbf{U}\in\mathbb{R}^{d_k\times r}$ | 候选低秩 correlation subspace |
| $\mathbf{H}_t\in\mathbb{R}^{r\times r}$ | 低维子空间中的相关结构 |
| $r$ | 候选 correlation rank，要求 $r\ll d_k$ |
