# The Proposed CNN-based CMTP-CNN Scheme (Paper Draft)

> **Document type**: Paste-ready Section III draft in IEEE Communications Letters style, aligned with Fu *et al.*, IEEE Commun. Lett., 2025 (DMISC), Section III.
> **Default checkpoint**: `models/cnn_improve_next/aug_spec_only/best_model.pth`.
> **Architecture figure**: [`figures/stp_cnn_architecture.png`](figures/stp_cnn_architecture.png) (generate with `python script/docs/plot_stp_cnn_architecture.py`).
> **Reference**: Y. Fu, Z. Zhao, J. Yan, and T. Q. S. Quek, “A Semantic Communication Scheme for Distributed Holographic-Type Communications With Multi-View Images,” *IEEE Commun. Lett.*, vol. 29, no. 5, pp. 878–882, May 2025.

---

## III. The Proposed CNN-based CMTP-CNN Scheme

In this section, the model structure of the proposed CNN-based CMTP-CNN model is designed to implement dual-BS spectral encoding and joint localization decoding. Then, the loss function and training strategy are introduced. Here, CMTP-CNN denotes the cooperative monostatic target-position convolutional neural network.

![CMTP-CNN architecture](figures/stp_cnn_architecture.png)

**Fig. 1.** The architecture of the proposed CNN-based CMTP-CNN model. (The convolution layers are parameterized by $(c,k,s)$, where $c$ is the number of channel outputs, $k$ is the kernel length, and $s$ is the stride; $\downarrow$ indicates down-sampling along the range axis.)

### A. The Architecture of Spectral Encoder $E(\cdot;\theta)$ at BS-$n$

A CNN-based spectral encoder is employed at each base station to map the complex range profile into a fixed-length semantic vector. As shown in Fig. 1, the encoders equipped at BS-$0$ and BS-$1$ are identical and share the parameters $\theta$. For BS-$n$, $n\in\{0,1\}$, let $\mathbf{x}_n\in\mathbb{C}^{L}$ denote the complex range profile after ROI cropping to $[0,4]\,\mathrm{m}$, where $L$ is the number of range bins. The encoder consists of a feature extraction block (FEB) to form real-valued inputs, a dual-BS shared spectral encoder (DSSE) composed of convolution blocks to extract the tensor-version semantic features $\mathbf{y}_n$, and a range attention pool (RAP) to map $\mathbf{y}_n$ into the semantic vector $\mathbf{s}_n$, i.e.,

$$
\mathbf{s}_n=E(\mathbf{x}_n;\theta).
$$

**1) The Architecture of FEB:**
The complex profile $\mathbf{x}_n$ is first split into real and imaginary parts to obtain a two-channel real-valued input

$$
\tilde{\mathbf{x}}_n=\bigl[\Re\{\mathbf{x}_n\},\,\Im\{\mathbf{x}_n\}\bigr]\in\mathbb{R}^{2\times L}.
$$

Please note that per-BS RMS normalization is not applied, so as to preserve the absolute spectral amplitude information for subsequent localization.

**2) The Architecture of DSSE:**
The intermediate feature of $\tilde{\mathbf{x}}_n$ is encoded by a stem convolution block followed by three residual convolution blocks with shared weights across BS-$0$ and BS-$1$. In particular, the stem employs $\mathrm{Conv1d}(2\to 64,\,k=7,\,s=2)$ with BatchNorm and ReLU, followed by $\mathrm{MaxPool1d}(k=3,\,s=2)$, which down-samples the range axis by approximately a factor of four. The first residual block keeps $64$ channels with stride $1$, while the second and third residual blocks raise the channel dimension to $128$ and $256$ with stride $2$, respectively. Each residual block consists of two $3\times 1$ convolutions, BatchNorm, ReLU, and a $1\times 1$ shortcut when the channel count or stride changes. The resulting semantic coded tensor is

$$
\mathbf{y}_n\in\mathbb{R}^{C\times L'},
$$

where $C=256$ and $L'\approx L/16$.

**3) The Architecture of RAP:**
$\mathbf{y}_n$ is compressed into a one-dimensional semantic vector by attention pooling along the range axis. Let $q(\cdot)$ denote a $1\times 1$ convolutional scoring network. The attention weights and the pooled vector are given by

$$
\boldsymbol{\alpha}_n=\mathrm{softmax}\bigl(q(\mathbf{y}_n)\bigr)\in\mathbb{R}^{1\times L'},
\qquad
\mathbf{s}_n=\sum_{\ell=1}^{L'}\alpha_{n,\ell}\,\mathbf{y}_{n,:,\ell}\in\mathbb{R}^{C}.
$$

In this way, RAP adaptively emphasizes range bins associated with target echoes while suppressing sidelobe- and noise-dominated regions, and yields $\mathbf{s}_n\in\mathbb{R}^{256}$ for each of BS-$0$ and BS-$1$.

### B. The Architecture of Joint Localization Decoder $D(\cdot;\zeta)$ at $R$

In this part, a joint localization decoder $D(\cdot;\zeta)$ is designed at the fusion center $R$ to recover the target planar coordinate from the dual-BS semantic vectors. As shown in Fig. 1, the decoder jointly processes $\mathbf{s}_0$ and $\mathbf{s}_1$ and directly regresses

$$
\hat{\mathbf{p}}=D(\mathbf{s}_0,\mathbf{s}_1;\zeta)=(\hat{x},\hat{y})\in\mathbb{R}^{2}.
$$

Unlike an image-reconstruction decoder, the present decoder does not employ deconvolutional up-sampling. Instead, late fusion followed by a multilayer perceptron (MLP) is used to explore the complementary information between BS-$0$ and BS-$1$. The detailed design is introduced as follows.

**1) The Architecture of Late Fusion Regression Head (LFRH):**
The main task of LFRH is to fuse the dual-BS semantic vectors and map them to planar coordinates. In particular, $\mathbf{s}_0$ and $\mathbf{s}_1$ are first concatenated as $\mathbf{s}=[\mathbf{s}_0;\mathbf{s}_1]\in\mathbb{R}^{512}$. Then, an MLP with one hidden layer of dimension $128$ is employed, where ReLU and dropout with probability $0.3$ are inserted between the two linear layers, i.e.,

$$
\hat{\mathbf{p}}=
\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{s}+\mathbf{b}_1)+\mathbf{b}_2.
$$

Please note that localization is performed in an end-to-end manner from dual-BS spectra to coordinates, without explicit single-BS range estimation followed by dual-circle intersection. The overall number of trainable parameters of CMTP-CNN is approximately $5.0\times 10^{5}$.

### C. Loss Function and Training Strategy

To minimize the difference between the predicted and ground-truth planar coordinates, a differentiable batch root-mean-square Euclidean distance (RMSE) is used as the loss function. For a mini-batch of size $B$, with prediction $\hat{\mathbf{p}}_i$, ground truth $\mathbf{p}_i$, and sample weight $w_i$,

$$
\mathcal{L}
=\sqrt{
\frac{\sum_{i=1}^{B} w_i\,\lVert\hat{\mathbf{p}}_i-\mathbf{p}_i\rVert_2^{2}}
{\sum_{i=1}^{B} w_i}
+\varepsilon
},
$$

where $\varepsilon>0$ is a numerical stabilizer. The weight $w_i$ is assigned according to the planar zone of the target: the center zone ($|x|,|y|\le 0.5\,\mathrm{m}$), side zones, and corner zones use weights $1$, $3$, and $3$, respectively, which can improve the fitting performance in geometrically adverse outer regions.

In Algorithm 1, an end-to-end training strategy is designed. The spectral encoder $E(\cdot;\theta)$ at BS-$0$/BS-$1$ and the joint localization decoder $D(\cdot;\zeta)$ at $R$ are trained in tandem by calculating $\mathcal{L}$. The Adam optimizer is employed, the learning rate is set as $5\times 10^{-5}$, the batch size is set as $128$, and the number of epochs is set as $E=100$, with early stopping when the validation loss does not improve for $P=15$ consecutive epochs. During training, uniform planar label jitter within $\pm 0.02\,\mathrm{m}$ and range-axis SpecAugment with probability $0.5$ are employed for data augmentation, together with CPI-domain amplitude scaling of $0.2$ and complex additive noise with standard deviation $0.02$ to enhance robustness to spectral fluctuations. The checkpoint with the best validation loss is retained for inference.

<table style="width:100%; border-collapse:collapse; border-top:2px solid #000; border-bottom:2px solid #000; margin:1em 0;">
<thead>
<tr><th style="text-align:left; border-bottom:1px solid #000; padding:0.4em 0.2em; font-weight:bold;">
Algorithm 1  The End-to-End Training Strategy of CMTP-CNN
</th></tr>
</thead>
<tbody>
<tr><td style="padding:0.35em 0.2em; vertical-align:top;">
<strong>1:</strong> Initialization: The model parameters $\theta^{(0)}$ and $\zeta^{(0)}$, the number of epochs $E$, and the early-stop patience $P$.
</td></tr>
<tr><td style="padding:0.35em 0.2em; vertical-align:top;">
<strong>2:</strong> For the $e$-th iteration, $1\le e\le E$
<ul style="margin:0.2em 0 0.2em 1.5em; padding:0;">
<li>Sample a mini-batch $\{(\mathbf{x}_0^{(i)},\mathbf{x}_1^{(i)},\mathbf{p}^{(i)})\}_{i=1}^{B}$ and apply data augmentation.</li>
<li>Update $\mathbf{s}_0,\mathbf{s}_1$ individually by $E(\cdot;\theta^{(e-1)})$ at BS-$0$ and BS-$1$.</li>
<li>Update $\hat{\mathbf{p}}$ jointly by $D(\cdot;\zeta^{(e-1)})$ at $R$.</li>
<li>Calculate $\mathcal{L}$ to update $\theta^{(e)}$ and $\zeta^{(e)}$.</li>
<li>If the validation loss does not improve for $P$ consecutive epochs, <strong>break</strong>.</li>
</ul>
</td></tr>
<tr><td style="padding:0.35em 0.2em; vertical-align:top;">
<strong>3:</strong> Output: The final training results of $E(\cdot;\theta)$ and $D(\cdot;\zeta)$.
</td></tr>
</tbody>
</table>
