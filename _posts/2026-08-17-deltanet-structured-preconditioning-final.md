---
layout: post
title: "从 DeltaNet 到结构化预条件：在稳定与并行约束下补回非对角曲率"
date: 2026-08-17 17:10:00
permalink: /blog/2026/deltanet-structured-preconditioning/
description: "从在线 ridge regression 重新理解 DeltaNet、MesaNet 与 Preconditioned DeltaNet，并讨论如何在 bounded stability 与 GPU parallelism 下补回 diagonal preconditioner 丢失的低秩相关性。"
tags: [DeltaNet, MesaNet, preconditioning, linear-attention, low-rank, optimization]
categories: [research-notes]
featured: false
giscus_comments: false
---

# 从 DeltaNet 到结构化预条件：在稳定与并行约束下补回非对角曲率

本文关心的不是“怎样给 DeltaNet 再加一个 low-rank 模块”，而是一个更具体的问题：**如果 DeltaNet 本质上在做在线回归，那么我们能否在保持数值稳定和 GPU 并行的前提下，比 diagonal preconditioning 多保留一点真正有用的二阶几何？**

我们目前的核心判断是：DeltaNet 没有显式处理 key space 的 curvature；[MesaNet](https://arxiv.org/abs/2506.05233) 用多步迭代 solver 处理完整 curvature；[Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100) 用 bounded diagonal correction 换取稳定性和 chunkwise parallelism。我们的方向不是回到 full inverse，而是研究 **diagonal correction 之后剩余的 non-diagonal error 是否集中在一个很小的子空间中，并只修正这些方向。**

相关的数值优化视角可以参考 [K-FAC](https://arxiv.org/abs/1503.05671)、[GGT / Full-Matrix AdaGrad](https://arxiv.org/abs/1806.02958)、[Shampoo](https://arxiv.org/abs/2506.03595) 和 [Sketchy](https://arxiv.org/abs/2302.03764)。这些工作共同提醒我们：对角二阶统计只能改变坐标轴尺度，不能表示旋转后的主曲率方向；但 full matrix 又往往过于昂贵。因此真正的问题是如何寻找 **结构化的 non-diagonal correction**。

## 1. 从优化视角重新看 DeltaNet

令 $S\in\mathbb R^{d_v\times d_k}$ 是 key 到 value 的线性 memory。给定到时刻 $t$ 为止的 key-value pairs，考虑 ridge regression

$$
S_t^*=\arg\min_S\frac12\sum_{i=1}^t\|Sk_i-v_i\|_2^2+\frac\lambda2\|S\|_F^2.
$$

记 $G_t=\lambda I+\sum_{i=1}^t k_i k_i^\top$，$C_t=\sum_{i=1}^t v_i k_i^\top$。对 $S$ 求导并令梯度为零，得到正规方程 $S_t^*G_t=C_t$，因此

$$
S_t^*=C_tG_t^{-1}.
$$

这就是后面所有讨论的起点。$C_t$ 记录“写入了什么”，而 $G_t$ 描述 key space 的二阶几何：$G_{ii}$ 是第 $i$ 个坐标的累计尺度，$G_{ij}$ 则描述第 $i,j$ 个坐标之间的耦合。

DeltaNet 并不显式求这个最优解。它只对当前 token 的损失 $\mathcal L_t(S)=\frac12\|Sk_t-v_t\|_2^2$ 做一次梯度下降。因为 $\nabla_S\mathcal L_t=(Sk_t-v_t)k_t^\top$，所以

$$
S_t=S_{t-1}+\beta_t(v_t-S_{t-1}k_t)k_t^\top.
$$

这里 $v_t-S_{t-1}k_t$ 决定“写什么”，$k_t$ 决定“沿哪个 key 方向修改 memory”。问题在于，所有方向只共享一个标量步长 $\beta_t$。如果 key space 接近各向同性，这种更新足够合理；如果 $G_t$ 很病态，那么不同方向理想的更新尺度应该明显不同。

因此从优化角度，DeltaNet 缺少的不是一个新的 gate，而是 $G_t$ 所描述的 curvature information。[Preconditioned DeltaNet](https://arxiv.org/abs/2604.21100) 正是从这一点出发：delta-rule recurrence 可以看成 least-squares 的一阶近似，而 preconditioning 的作用是让不同 key 方向的更新更接近这个二阶问题的真实几何。

## 2. 为什么不直接使用 full $G_t^{-1}$

如果完全不考虑效率，最自然的目标当然是使用 $G_t^{-1}$。在在线 DeltaNet 中，用 Sherman--Morrison 可以把 exact ridge solution 写成一个 delta-style update。若记 $P_{t-1}=G_{t-1}^{-1}$，则理想 write key 为 $\widetilde k_t=P_{t-1}k_t/(1+k_t^\top P_{t-1}k_t)$，并有 $S_t=S_{t-1}+(v_t-S_{t-1}k_t)\widetilde k_t^\top$。在相应假设下，这个递推精确恢复 $S_t^*=C_tG_t^{-1}$；这正是 PDN 论文 Theorem 3.1 的理论出发点。

问题首先是状态开销。$G_t\in\mathbb R^{d_k\times d_k}$，显式维护 $G_t$ 或 $P_t=G_t^{-1}$ 都需要 $O(d_k^2)$ 状态。更麻烦的是 inverse 的 Sherman--Morrison 更新依赖 $P_{t-1}$ 的完整矩阵状态，很难写成 PDN 所需要的简单 elementwise scan。

第二个问题是 GPU。DeltaNet 的效率来自 chunkwise parallel form，其主要工作最终可以落到 GEMM 上；full inverse 或逐 token 的矩阵分解会把计算重新变成小矩阵、强依赖、低 arithmetic intensity 的操作。PDN 论文也明确指出，exact inverse-Gram recurrence 无法得到他们需要的 chunk-local affine transition。

第三个问题是数值稳定。$G_t^{-1}$ 会放大小特征值方向；如果 $G_t$ 条件数很大，精确 inverse 虽然在数学上正确，却未必是一个好训练算子。

[MesaNet](https://arxiv.org/abs/2506.05233) 的处理方式是避免显式 inverse。用与本文统一的记号，它要计算 $x_t=G_t^{-1}q_t$，等价于解线性系统 $G_tx_t=q_t$。MesaNet 用 conjugate gradient 迭代求 $x_t$，其中最主要的矩阵向量乘 $G_tp$ 可以利用 linear-attention/chunkwise 结构实现。因此它保留了更完整的 curvature information，但代价是每次需要多步 solver compute，并且论文专门讨论了 CG 在训练中的数值稳定问题。

一个对我们很有启发的细节是：MesaNet 的 CG 从 diagonal inverse 初始化。也就是说，它先用 $\operatorname{Diag}(G_t)^{-1}q_t$ 得到一个便宜的近似，再让 CG 继续修正 diagonal 没有解决的部分。这个视角后面会直接导向我们的 $U$。

PDN 选择的是另一端：不运行多步 solver，而是只维护 $G_t$ 的 diagonal 二阶统计。设

$$
A_t=\alpha_t^P A_{t-1}+\beta_t^P(k_t\odot k_t),
$$

其中 $A_t\in\mathbb R^{d_k}$ 可以看成在线的 diagonal Gram statistics。这个递推是逐元素 affine recurrence，因此天然可以做 prefix scan；这也是它比 full $G_t$ 更适合 GPU 的根本原因。

但 PDN 并没有直接令 write scaling 为 $A_t^{-1}$。论文观察到 $A_t$ 的经验分布具有明显的 heavy right tail，并在 log-spaced histogram 中呈现近似 log-normal 结构。小 $A_{t,j}$ 在直接 reciprocal 后会被放成非常大的数，因此作者主动放弃 exact inverse，改成 $r_t=\log A_t-\mu$、$s_t=r_t/(1+|r_t|)$，再令 $B_t=\exp[-\log(x)s_t]$。于是每一个坐标都有 $1/x\le B_{t,j}\le x$。

![Preconditioned DeltaNet Figure 2: diagonal Gram statistics before and after bounded squashing](preconditioned-deltanet-figure2-heavy-tail.png)

*图：Preconditioned DeltaNet 的 Figure 2。左图展示 $A_t$ 在 log-spaced buckets 下的长尾 / 近似 log-normal 分布，右图展示经过 bounded mapping 后的 $B_t$。来源：[Preconditioned DeltaNet, Figure 2](https://arxiv.org/abs/2604.21100)。*

这一点非常重要。PDN 并不是认为 $A_t^{-1}$ 不对，而是认为 **作为神经网络中的 recurrence，preconditioner 首先必须稳定，然后才谈是否更接近 inverse。** 它保留了 inverse 的单调性：$A_{t,j}$ 越大，$B_{t,j}$ 越小；但不允许 scaling 无限增大或无限缩小。

如果写 $P_t^{\mathrm{diag}}=\operatorname{Diag}(B_t)$，那么 PDN 使用 $\widetilde k_t=P_t^{\mathrm{diag}}k_t$。对于 normalized key，若 $P_t^{\mathrm{diag}}$ 的谱被限制在 $[1/x,x]$，则 $k_t^\top P_t^{\mathrm{diag}}k_t\le x$。因此控制 $\beta_tx\le2$ 就可以控制关键 rank-one transition 的非平凡特征值。这解释了为什么 bounded mapping 不只是一个经验上的 clipping trick，而是和 recurrence stability 直接相连。

所以到这里，full inverse、MesaNet 和 PDN 形成了一个很清楚的谱系：full inverse 最接近精确 ridge solver，但难以并行且可能不稳定；MesaNet 用 iterative solver 避免 inverse-state recurrence，但需要多步计算；PDN 只保留 diagonal curvature，并进一步把 inverse 映射限制在稳定区间，从而得到 scan-friendly、GPU-friendly 的更新。

## 3. diagonal 到底丢掉了什么

PDN 的 diagonal approximation 并不是没有依据。论文在 pretrained DeltaNet 中观察到 full key Gram 与 diagonal Gram 的 eigenspectrum 高度相关，而且主要 eigenvectors 相对 axis-aligned。这说明在它们观察的模型中，diagonal 是一个合理的效率折中。

我们的 claim 因此不能是“diagonal approximation 错了”。更准确的问题是：**在 diagonal 已经处理掉主要 coordinate-wise scale 后，是否仍然存在少数对 preconditioning 很重要的 non-axis-aligned directions？**

最简单的例子是

$$
G=\begin{bmatrix}1&0.99\\0.99&1\end{bmatrix}.
$$

它的 diagonal approximation 就是 $I$，看起来 condition number 为 $1$；但 full $G$ 的 eigenvalues 是 $1.99$ 和 $0.01$，真实 condition number 为 $199$。真正的高、低曲率方向分别是 $(1,1)/\sqrt2$ 和 $(1,-1)/\sqrt2$。单纯 coordinate-wise scaling 根本看不到这两个旋转后的方向。

不过我们也不能直接拿 raw off-diagonal $G_{ij}$ 来判断 correlation。因为 $G_{ij}$ 的绝对大小受到两个坐标本身尺度的影响。例如

$$
G=\begin{bmatrix}100&9\\9&1\end{bmatrix}.
$$

这里 off-diagonal 是 $9$。它到底算大还是小？因为 $\sqrt{G_{11}G_{22}}=10$，真正的 normalized correlation 是 $9/10=0.9$。相反，矩阵 $\begin{bmatrix}0.01&0.009\\0.009&0.01\end{bmatrix}$ 的 off-diagonal 只有 $0.009$，但 normalized correlation 同样是 $0.9$。

因此我们不希望让新的 low-rank part 再去学习 PDN 已经处理的 marginal scale。令 $D_t=\operatorname{Diag}(\operatorname{diag}(G_t))$，定义

$$
R_t=D_t^{-1/2}G_tD_t^{-1/2},\qquad G_t=D_t^{1/2}R_tD_t^{1/2}.
$$

这是恒等式，不是近似。$D_t$ 保留“每个坐标有多大”，而 $R_t$ 把这些 marginal scale 去掉，只保留剩余的 correlation geometry。更关键的是 $G_t^{-1}=D_t^{-1/2}R_t^{-1}D_t^{-1/2}$，所以我们从来没有丢掉尺度；我们只是把 scale correction 和 correlation correction 分开。

因为 $\operatorname{diag}(R_t)=\mathbf1$，所以可以写 $R_t=I+E_t$。这里 $E_t=R_t-I$ 可以理解为 **diagonal normalization 之后剩下的全部二阶误差**。如果 $E_t\approx0$，那么 PDN 已经足够；如果 $E_t$ 只在少数方向显著，那么就出现了我们真正想利用的结构。

我们目前的核心经验假设是

$$
R_t-I\approx U\Lambda_tU^\top,\qquad U\in\mathbb R^{d_k\times r},\quad r\ll d_k.
$$

这意味着大多数 normalized directions 已经被 diagonal preconditioner 处理到接近 curvature $1$，只有一个低维子空间仍然明显偏离 identity。$U$ 表示这些方向，$\Lambda_t$ 表示这些方向当前偏离 $1$ 的程度。

这个假设还有一个很直接的 solver 解释。MesaNet 的 diagonal initial guess 在 normalized coordinates 中相当于先假设 $R_t=I$。令 normalized linear system 为 $(I+E_t)y=b$，取 diagonal initial guess $y^{(0)}=b$，则 residual 为 $r^{(0)}=b-(I+E_t)b=-E_tb$。如果 $E_t\approx U\Lambda_tU^\top$，则 $r^{(0)}\approx-U\Lambda_tU^\top b$，所以 diagonal solver 剩下的 error 主要落在 $\operatorname{span}(U)$ 中。

这给了 $U$ 一个比“low-rank compression”更准确的解释：**$U$ 表示 diagonal preconditioning 之后仍然反复需要 solver 修正的方向。**

## 4. $U$ 应该追踪什么，而不是怎么随便做一个 PCA

如果 $R_t=Q\operatorname{Diag}(\rho_i)Q^\top$，那么 exact correlation inverse 是 $Q\operatorname{Diag}(1/\rho_i)Q^\top$。但我们已经从 PDN 学到，实际模型不应该追求无界的 $1/\rho_i$。令 $\phi(\rho)$ 表示一个 bounded inverse-like mapping，例如满足 $\phi(1)=1$ 且 $1/x_c\le\phi(\rho)\le x_c$。

那么理想的 correlation correction 是 $P_t^{\mathrm{corr},*}=Q\operatorname{Diag}(\phi(\rho_i))Q^\top$。如果我们只允许一个 rank-$r$ correction $I+UCU^\top$，真正应该逼近的是 $P_t^{\mathrm{corr},*}-I$，而不是 $R_t$ 本身。因此理想的 $U$ 应该优先选择 $|\phi(\rho_i)-1|$ 最大的 eigen-directions。

这和普通 PCA 有本质区别。PCA 只会偏好大的 $\rho_i$；preconditioning 同样关心很小的 $\rho_i$。例如 $\rho_1=10$、$\rho_2=0.01$ 时，第二个方向虽然不是最大 variance direction，却可能是 inverse 中最危险、最需要修正的方向。

在 $\rho\approx1$ 附近，bounded inverse mapping 可以局部写成 $\phi(\rho)-1\approx-c(\rho-1)$。因此一个简单的 surrogate 是追踪 $(R_t-I)^2$ 的 dominant eigenspace，因为它对 eigenvector $q_i$ 的 eigenvalue 是 $(\rho_i-1)^2$，会同时强调 $\rho_i\gg1$ 和 $\rho_i\ll1$ 的异常方向。

这给出我们目前对 $U$ 的几种设计。

**第一种，也是最应该先做的版本，是 static learned $U$。** 每个 head 学一个 $U\in\mathbb R^{d_k\times r}$，训练时作为普通参数更新，一个 sequence 内不随 token 变化。我们通过正交正则或低频 retraction 使 $U^\top U\approx I$。它的优势非常直接：对整个 chunk 的 normalized keys $\bar K$，所有低维坐标一次 GEMM 就可以得到 $H=\bar K U$。之后动态的只有 $r$ 维 statistic，因此保持 GEMM + scan 的 GPU 结构。

**第二种是 preconditioning-aware spectral warm start。** 离线收集真实 $R_t$，做 oracle eigendecomposition，不按最大 $\rho_i$ 选择 PCA directions，而按 $|\phi(\rho_i)-1|$ 选方向。这个版本不用于部署，而用于回答一个关键问题：如果我们把“最需要修正的 directions”直接给模型，rank $4/8/16$ 到底够不够？它也是 static learned $U$ 最自然的初始化。

**第三种是 chunk-wise slow $U$。** 如果实验发现 basis 确实随上下文缓慢旋转，可以在一个 chunk 内固定 $U_c$，只在 chunk boundary 更新。一个最简单的 subspace update 是

$$
U_{c+1}=\operatorname{qf}\!\left(U_c+\eta_c(R_c-I)^2U_c\right).
$$

这里 $\operatorname{qf}$ 表示低频正交化。关键是我们不需要显式形成 $R_c\in\mathbb R^{d_k\times d_k}$。若一个 chunk 的 normalized key matrix 为 $Z_c\in\mathbb R^{C\times d_k}$，则对任意 $V\in\mathbb R^{d_k\times r}$ 都有 $R_cV=Z_c^\top(Z_cV)/C$。因此先算 $V=R_cU_c-U_c$，再算 $R_cV-V$，就得到了 $(R_c-I)^2U_c$。整个过程的主体仍然是 $Z_cU_c$、$Z_c^\top(Z_cU_c)$ 这样的 GEMM，而不是逐 token SVD。

**第四种是 sketch / Frequent Directions。** [Sketchy](https://arxiv.org/abs/2302.03764) 等 full-matrix adaptive optimization 工作说明，可以用 $O(d_kr)$ 状态维护 covariance 的低秩 sketch。这提供了另一条动态 basis 路线，但周期性的 shrink、QR 或 SVD 会增加实现复杂度。因此我们暂时把它放在 static / chunk-wise 方案之后。

我们暂时不考虑 token-wise Oja 或 token-wise PCA。原因不是这些算法没有理论，而是它们引入 $U_t\rightarrow U_{t+1}$ 的逐 token matrix dependency，还需要频繁正交化。这和我们的第一约束——GPU-friendly——直接冲突。

## 5. $c_t$：方向固定以后，每个方向应该修正多少

$U$ 解决的是“修正哪些方向”，$c_t$ 解决的是“这些方向当前修正多少”。这两个变量应该有不同时间尺度：$U$ 是 slow basis，$c_t$ 是 fast spectrum。

先用 diagonal statistics 把 key 做尺度归一化，记 $\bar k_t=D_t^{-1/2}k_t$；实际实现中这里不会直接使用无界的 $D_t^{-1/2}$，而会和 PDN 的 bounded diagonal scaling 对齐。然后计算 $h_t=U^\top\bar k_t$。如果 $U$ 近似是 $R_t$ 的相关方向，那么第 $j$ 个方向上的 normalized curvature 就可以通过 $h_{t,j}^2$ 的在线统计估计。一个简单的递推是 $a_t=\alpha_t^Ca_{t-1}+\beta_t^C(h_t\odot h_t)$，并配合相同权重的 mass recurrence 做归一化。

如果 normalization 完全正确且没有额外 correlation，那么基准 curvature 是 $1$。因此理想情况下，第 $j$ 个方向的 inverse scale 是 $1/a_{t,j}$。但我们不直接用 reciprocal，而和 PDN 完全一样，把它变成 bounded inverse。可以令 $r_{t,j}=\log a_{t,j}-\mu_c$、$s_{t,j}=r_{t,j}/(1+|r_{t,j}|)$，再令 $b_{t,j}^{\mathrm{corr}}=\exp[-\log(x_c)s_{t,j}]$，最后定义 $c_{t,j}=b_{t,j}^{\mathrm{corr}}-1$。

这样 $1+c_{t,j}$ 永远落在 $[1/x_c,x_c]$。当 $a_{t,j}>1$ 时，该方向 curvature 偏大，所以 $c_{t,j}<0$，应该抑制；当 $a_{t,j}<1$ 时，该方向 curvature 偏小，所以 $c_{t,j}>0$，应该增强；当 $a_{t,j}=1$ 时，$c_{t,j}=0$，不需要额外 correction。

因此我们的 low-rank part 并没有重新发明一套稳定化机制。它只是把 PDN 的核心思想从坐标轴推广到 $U$ 所定义的 correlation directions：**PDN 对 coordinate curvature 做 bounded inverse，我们对 residual correlation curvature 做 bounded inverse。**

## 6. 最终递推和 GPU 形式

记 $B_t^{\mathrm{diag}}\succ0$ 为 PDN 已有的 bounded diagonal preconditioner，并令 $U^\top U=I$。我们目前最干净的组合是

$$
P_t=(B_t^{\mathrm{diag}})^{1/2}\left[I+U\operatorname{Diag}(c_t)U^\top\right](B_t^{\mathrm{diag}})^{1/2}.
$$

最终 write key 为 $\widetilde k_t=P_tk_t$，直接放回 Gated DeltaNet / PDN 的原递推：

$$
S_t=\alpha_tS_{t-1}+\beta_t\left(v_t-\alpha_tS_{t-1}k_t\right)\widetilde k_t^\top.
$$

当 $c_t=0$ 时，correlation correction 退化为 identity，模型回到 PDN。这个形式还有一个重要好处：如果 $B_t^{\mathrm{diag}}$ 的谱在 $[1/x_d,x_d]$，而 $I+U\operatorname{Diag}(c_t)U^\top$ 的谱在 $[1/x_c,x_c]$，那么最终 $P_t$ 的谱被限制在 $[1/(x_dx_c),x_dx_c]$。因此我们仍然可以沿用 PDN 的 bounded-spectrum stability 逻辑，而不是为了 non-diagonal correction 重新引入一个不可控的 inverse。

static $U$ 时，GPU 路径也很直接。一个 chunk 内，先用 PDN 的 affine scan 得到 diagonal statistics；把 key 按 diagonal scale 做 elementwise normalization 得到 $\bar K$；然后一次 GEMM 计算 $H=\bar K U$；对 $H\odot H$ 做 $r$ 维 affine scan 得到 $c_t$；再通过 $(C\odot H)U^\top$ 这样的 GEMM 把 low-rank correction 投回 $d_k$ 维。额外核心计算是 $O(Cd_kr)$，没有 $O(d_k^2)$ 状态，也没有逐 token 矩阵求逆或分解。

所以我们现在的主要困难已经很集中：不是怎样再设计一个复杂 preconditioner，而是 **$U$ 能否在极低的硬件代价下，稳定地覆盖 diagonal PDN 真正遗漏的 preconditioning directions。**

## 7. 当前最需要验证的三个问题

第一，$R_t-I$ 是否真的显著。如果 $R_t\approx I$，那么 PDN 已经基本完成了 whitening，non-diagonal correction 没有意义。

第二，真正需要修正的 spectral error 是否低秩。这里不应只看 PCA explained variance，而应该看 rank-$r$ 子空间能解释多少 $|\phi(\rho_i)-1|$。我们关心的是 preconditioning error，而不是 covariance energy。

第三，这些 directions 是否稳定。最直接的 oracle 实验是同时比较三种 rank-$r$ basis：按最大 $\rho_i$ 选择的 PCA basis、按最大 $|\rho_i-1|$ 选择的 residual basis，以及按最大 $|\phi(\rho_i)-1|$ 选择的 preconditioning-aware basis。如果第三种在很小的 $r$ 下已经明显逼近 full bounded preconditioner，并且这些方向跨 chunk / sequence 相对稳定，那么 static learned $U$ 就有充分依据；如果方向缓慢变化，再升级到 chunk-wise slow $U$；如果谱根本不低秩，则应该转向 block / Kronecker 等结构，而不是继续强行做 low rank。

目前我们的 claim 可以压缩成一句话：

> **DeltaNet 忽略 curvature；PDN 用 bounded diagonal curvature 换取稳定和并行；我们的目标是在同样的稳定与 GPU 约束下，只补回 diagonal 之后仍然显著的低秩 non-diagonal preconditioning error。**

这比“给 PDN 增加一个 $U$”更准确。$U$ 只是实现这个 claim 的方向变量，$c_t$ 是动态强度变量；真正的研究问题始终是：**哪些二阶几何值得在固定的 memory、stability 和 hardware budget 下被保留下来。**
