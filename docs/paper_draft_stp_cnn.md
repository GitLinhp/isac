# The Proposed CNN-based CMTP Scheme (Paper Draft)

> **Document type**: Paste-ready draft in IEEE Communications Letters style, aligned with Fu *et al.*, IEEE Commun. Lett., 2025 (DMISC), Sections III–IV.
> **Includes**: Prototype Signal Processing Procedure (TX / RX / fusion center) + Section III CMTP architecture (paste-ready LaTeX) + Section IV Experimental Evaluation.
> **Default checkpoint**: `models/cnn_improve_next/aug_spec_only/best_model.pth`.
> **Architecture figure**: [`figures/stp_cnn_architecture.png`](figures/stp_cnn_architecture.png) (generate with `python script/docs/plot_stp_cnn_architecture.py`).
> **Reference**: Y. Fu, Z. Zhao, J. Yan, and T. Q. S. Quek, “A Semantic Communication Scheme for Distributed Holographic-Type Communications With Multi-View Images,” *IEEE Commun. Lett.*, vol. 29, no. 5, pp. 878–882, May 2025.

---

## Signal Processing Procedure (Prototype)

> Paste-ready LaTeX for `\subsection{Signal Processing Procedure}`. Aligns with the dual-BS cooperative monostatic prototype (USRP X410 + Echotimer). No CFAR. Receiver range spectrum uses one equation per step; fusion center follows single-model CMTP ($f$, $\Theta$, $\mathbf{r}_n$, $\mathbf{s}_n$, $\hat{\mathbf{p}}$, ROI $[0,4]\,\mathrm{m}$).

```latex
\subsection{Signal Processing Procedure}
The baseband signal processing can be divided into transmitter, receiver and data fusion center. The detailed signal processing procedures are shown as follows.

At the transmitters, a pre-stored bitstream is loaded and processed through QAM mapping, resource-grid mapping, inverse fast Fourier transform (IFFT), and cyclic-prefix (CP) insertion to form the transmitted OFDM baseband waveform. Each coherent processing interval (CPI) comprises several consecutive OFDM symbols. The baseband signal is then upconverted to radio frequency (RF) and radiated over the air. In the prototype, two quasi-monostatic base stations (BS-$0$ and BS-$1$) employ USRP X410 radios with Echotimer-based synchronization to maintain a stable TX/RX phase relationship across CPIs.

At the receivers, the received continuous-time signal $y(t)$ is processed in three steps---OFDM demodulation, least-squares (LS) channel estimation, and range-spectrum computation---to obtain the complex range spectrum $\mathbf{r}$. First, OFDM demodulation (CP removal and FFT) yields the frequency-domain observation
\begin{equation}
Y_{m,n}=\mathrm{Demod}\{y(t)\}_{m,n},
\end{equation}
where $m$ and $n$ index the sensing symbol and subcarrier, respectively. Second, LS channel estimation with the known transmit reference $X_{m,n}$ gives
\begin{equation}
\hat{H}_{m,n}=\frac{Y_{m,n}}{X_{m,n}}.
\end{equation}
Third, Blackman--Harris windowing, zero-padding to length $N_{\mathrm{FFT}}=ZN$, IFFT, and CPI integration are compactly written as
\begin{equation}
\mathbf{r}=\sum_{m=0}^{M_{\mathrm{CPI}}-1}
\mathbf{F}_{N_{\mathrm{FFT}}}^{-1}
\big(
\mathbf{w}\odot
[\hat{H}_{m,0},\ldots,\hat{H}_{m,N-1},\mathbf{0}_{1\times(N_{\mathrm{FFT}}-N)}]^{\mathsf{T}}
\big),
\end{equation}
where $\mathbf{F}_{N_{\mathrm{FFT}}}^{-1}$ is the $N_{\mathrm{FFT}}$-point IFFT, $\mathbf{w}$ is the window, and $\odot$ denotes the Hadamard product. The physical range of the $\ell$-th bin is $R_\ell=\ell\Delta R$ with resolution $\Delta R=C/(2N_{\mathrm{FFT}}\Delta f)$. The integrated spectrum is finally cropped to the ROI $[0,4]\,\mathrm{m}$ and uploaded to the data fusion center; the dual-BS ROI profiles are denoted by $\mathbf{r}_0$ and $\mathbf{r}_1$.

At the data fusion center, the dual-BS ROI spectra are jointly processed by the proposed CMTP to recover the target planar coordinate:
\begin{equation}
\hat{\mathbf{p}}=f(\mathbf{r}_0,\mathbf{r}_1;\Theta)=(\hat{x},\hat{y})\in\mathbb{R}^{2},
\end{equation}
where $\Theta$ denotes the trainable parameters. Unlike conventional pipelines that estimate per-BS ranges and then intersect dual-station circles, CMTP performs end-to-end localization from spectra to coordinates; the network architecture is detailed in Section~\ref{sec:CMTP scheme}.

In summary, the transmitter generates synchronized OFDM waveforms, each receiver extracts the ROI range profile, and the fusion center executes CMTP to output the target location in the sensing plane.
```

---

## III. The Proposed CNN-based CMTP Scheme

> Paste-ready LaTeX for `\section{The Proposed CNN-based CMTP Scheme}`. Overview (with preprocessing) then modules CB / ResBlock / RAP / MLP. Architecture figure: [`figures/stp_cnn_architecture.png`](figures/stp_cnn_architecture.png).

```latex
\section{The Proposed CNN-based CMTP Scheme}
\label{sec:CMTP scheme}
In this section, the architecture of the proposed CNN-based CMTP is presented. Then, the loss function and training strategy are introduced. Here, CMTP denotes the cooperative monostatic target-position convolutional neural network, which maps dual-BS ROI range spectra to the target planar coordinate in an end-to-end manner.

\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{figures/stp_cnn_architecture.png}
\caption{The architecture of the proposed CNN-based CMTP model. (The convolution layers are parameterized by $(c,k,s)$, where $c$ is the number of channel outputs, $k$ is the kernel length, and $s$ is the stride; $\downarrow$ indicates down-sampling along the range axis.)}
\Description{Block diagram of CMTP with dual-BS shared CB and ResBlocks, range attention pooling (RAP), and an MLP regression head.}
\label{fig:cmtp-arch}
\end{figure}

\subsection{The Architecture of CMTP}
As shown in Fig.~\ref{fig:cmtp-arch}, CMTP is a single CNN that maps dual-BS ROI range spectra to the planar coordinate $\hat{\mathbf{p}}=(\hat{x},\hat{y})$ end-to-end. For BS-$n$, $n\in\{0,1\}$, let $\mathbf{r}_n\in\mathbb{C}^{L}$ denote the complex range profile after ROI cropping to $[0,4]\,\mathrm{m}$, where $L$ is the number of range bins. As a preprocessing step, each complex profile is split into real and imaginary parts to form a two-channel real-valued input
\begin{equation}
\tilde{\mathbf{r}}_n=\bigl[\Re\{\mathbf{r}_n\},\,\Im\{\mathbf{r}_n\}\bigr]\in\mathbb{R}^{2\times L}.
\end{equation}
Please note that per-BS RMS normalization is not applied, so as to preserve the absolute spectral amplitude information for subsequent localization. The network then applies a shared-weight cascade of a convolution block (CB), three residual blocks (ResBlocks), and range attention pooling (RAP) to each station in parallel, followed by concatenation and an MLP regression head, i.e.,
\begin{equation}
\hat{\mathbf{p}}=f(\mathbf{r}_0,\mathbf{r}_1;\Theta),
\end{equation}
where $\Theta$ collects all trainable parameters of CMTP. The detailed architectures of CB, ResBlock, RAP, and MLP are introduced as follows.

1) \textit{The Architecture of CB:}
The intermediate feature of $\tilde{\mathbf{r}}_n$ is first extracted by CB along the range axis. In particular, CB employs $\mathrm{Conv1d}(2\to 64,\,k=7,\,s=2)$ with BatchNorm and ReLU, followed by $\mathrm{MaxPool1d}(k=3,\,s=2)$, which down-samples the range axis by approximately a factor of four.

2) \textit{The Architecture of ResBlock:}
The semantic coded tensor is then generated by refining the CB features through three ResBlocks with shared weights across BS-$0$ and BS-$1$. In particular, ResBlock$_1$ keeps $64$ channels with stride $1$, while ResBlock$_2$ and ResBlock$_3$ raise the channel dimension to $128$ and $256$ with stride $2$, respectively. Each ResBlock consists of two $3\times 1$ convolutions, BatchNorm, ReLU, and a $1\times 1$ shortcut when the channel count or stride changes. The resulting feature tensor is
\begin{equation}
\mathbf{y}_n\in\mathbb{R}^{C\times L'},
\end{equation}
where $C=256$ and $L'\approx L/16$.

3) \textit{The Architecture of RAP:}
$\mathbf{y}_n$ is compressed into a one-dimensional feature vector by RAP along the range axis. Let $q(\cdot)$ denote a $1\times 1$ convolutional scoring network. The attention weights and the pooled vector are given by
\begin{equation}
\boldsymbol{\alpha}_n=\mathrm{softmax}\bigl(q(\mathbf{y}_n)\bigr)\in\mathbb{R}^{1\times L'},
\qquad
\mathbf{s}_n=\sum_{\ell=1}^{L'}\alpha_{n,\ell}\,\mathbf{y}_{n,:,\ell}\in\mathbb{R}^{C}.
\end{equation}
In this way, RAP adaptively emphasizes range bins associated with target echoes while suppressing sidelobe- and noise-dominated regions, and yields $\mathbf{s}_n\in\mathbb{R}^{256}$ for each of BS-$0$ and BS-$1$.

4) \textit{The Architecture of MLP:}
The main task of MLP is to fuse the dual-BS feature vectors and map them to planar coordinates. In particular, $\mathbf{s}_0$ and $\mathbf{s}_1$ are first concatenated as $\mathbf{s}=[\mathbf{s}_0;\mathbf{s}_1]\in\mathbb{R}^{512}$. Then, an MLP with one hidden layer of dimension $128$ is employed, where ReLU and dropout with probability $0.3$ are inserted between the two linear layers, i.e.,
\begin{equation}
\hat{\mathbf{p}}=
\mathbf{W}_2\,\mathrm{ReLU}(\mathbf{W}_1\mathbf{s}+\mathbf{b}_1)+\mathbf{b}_2.
\end{equation}
Please note that localization is performed end-to-end from dual-BS spectra to coordinates, without explicit single-BS range estimation followed by dual-circle intersection. The overall number of trainable parameters of CMTP is approximately $5.0\times 10^{5}$.

\subsection{Loss Function and Training Strategy}
To minimize the difference between the predicted and ground-truth planar coordinates, a differentiable batch root-mean-square Euclidean distance (RMSE) is used as the loss function. For a mini-batch of size $B$, with prediction $\hat{\mathbf{p}}_i$, ground truth $\mathbf{p}_i$, and sample weight $w_i$,
\begin{equation}
\mathcal{L}
=\sqrt{
\frac{\sum_{i=1}^{B} w_i\,\lVert\hat{\mathbf{p}}_i-\mathbf{p}_i\rVert_2^{2}}
{\sum_{i=1}^{B} w_i}
+\varepsilon
},
\end{equation}
where $\varepsilon>0$ is a numerical stabilizer. The weight $w_i$ is assigned according to the planar zone of the target: the center zone ($|x|,|y|\le 0.5\,\mathrm{m}$), side zones, and corner zones use weights $1$, $3$, and $3$, respectively, which can improve the fitting performance in geometrically adverse outer regions.

In Algorithm~\ref{alg:cmtp}, an end-to-end training strategy is designed. All parameters $\Theta$ of CMTP are trained jointly by calculating $\mathcal{L}$. The Adam optimizer is employed, the learning rate is set as $5\times 10^{-5}$, the batch size is set as $128$, and the number of epochs is set as $E=100$, with early stopping when the validation loss does not improve for $P=15$ consecutive epochs. During training, uniform planar label jitter within $\pm 0.02\,\mathrm{m}$ and range-axis SpecAugment with probability $0.5$ are employed for data augmentation, together with CPI-domain amplitude scaling of $0.2$ and complex additive noise with standard deviation $0.02$ to enhance robustness to spectral fluctuations. The checkpoint with the best validation loss is retained for inference.

\begin{algorithm}[!htbp]
\caption{The End-to-End Training Strategy of CMTP}
\label{alg:cmtp}
\begin{algorithmic}[1]
\State \textbf{Initialization:}
The model parameters $\Theta^{(0)}$, the number of epochs $E$, and the early-stop patience $P$. For the $e$-th iteration, $1\le e\le E$:
\begin{itemize}
\item Sample a mini-batch $\{(\mathbf{r}_0^{(i)},\mathbf{r}_1^{(i)},\mathbf{p}^{(i)})\}_{i=1}^{B}$ and apply data augmentation.
\item Compute $\hat{\mathbf{p}}=f(\mathbf{r}_0,\mathbf{r}_1;\Theta^{(e-1)})$.
\item Calculate $\mathcal{L}$ to update $\Theta^{(e)}$. Early-stop if the validation loss does not improve for $P$ consecutive epochs.
\end{itemize}
\State \textbf{Output:} The final trained CMTP $f(\cdot;\Theta)$.
\end{algorithmic}
\end{algorithm}
```

---

## IV. Experimental Evaluation

> Paste-ready LaTeX for `\section{Experimental Evaluation}` with two subsections: Dataset Construction and Experimental Results. Aligns with Fu *et al.*, IEEE Commun. Lett., 2025, Section IV. Figures under [`figures/`](figures/). Metrics from `out/cooperative_monostatic/methods_compare/`.

```latex
\section{Experimental Evaluation}
In this section, we first describe the construction of the cooperative monostatic measurement dataset, and then evaluate the performance of the proposed CMTP-CNN scheme against classical subspace baselines.

\subsection{Dataset Construction}
The measurement campaign is conducted on a dual-BS cooperative monostatic ISAC prototype as described in Section~\ref{sec:prototype system design}. As shown in Fig.~\ref{fig:scene}, BS-$0$ and BS-$1$ are placed at $(0,-2)\,\mathrm{m}$ and $(-2,0)\,\mathrm{m}$, respectively, and each station operates in a quasi-monostatic configuration with a small TX/RX antenna offset. A single static target (metal reflector) is moved over the sensing plane $[-1,1]\,\mathrm{m}\times[-1,1]\,\mathrm{m}$. To characterize geometrically diverse localization conditions, the plane is divided into center zones ($|x|,|y|\le 0.5\,\mathrm{m}$), side zones, and corner zones, which are also used later for weighted training and regional metrics.

During the measurement campaign, both base stations transmit and receive synchronized OFDM waveforms. BS-$0$ and BS-$1$ operate at carrier frequencies of $6.0\,\mathrm{GHz}$ and $3.5\,\mathrm{GHz}$, respectively. The OFDM configuration employs a subcarrier spacing of $120\,\mathrm{kHz}$ with $2048$ subcarriers, yielding an effective bandwidth of $245.76\,\mathrm{MHz}$ and a range resolution of approximately $0.61\,\mathrm{m}$. Each coherent processing interval (CPI) comprises $M_{\mathrm{CPI}}=4$ consecutive OFDM symbols that are integrated at the receiver.

At each target location, a measurement session records synchronized CPI frames from both base stations. For each CPI, the receiver extracts the complex range profile, which is cropped to the ROI $[0,4]\,\mathrm{m}$ and stored together with the ground-truth coordinate $\mathbf{p}=(x,y)$. The dual-BS ROI spectra $\{\mathbf{r}_0,\mathbf{r}_1\}$ and the label $\mathbf{p}$ constitute one training / evaluation sample for CNN-based CMTP. Two independent measurement campaigns are conducted on two different days under the same geometry and processing pipeline. Each campaign covers $217$ distinct planar coordinates with up to $50$ CPI frames per location. The Day-1 dataset is partitioned into training and validation subsets with a validation ratio of $0.4$ in a session-wise and region-stratified manner: all CPI frames recorded at the same target location are assigned exclusively to either the training or the validation subset, and the split is performed independently within each planar zone. CMTP is trained on the training subset following Section~\ref{sec:CMTP scheme} and Algorithm~\ref{alg:cmtp}, with early stopping and checkpoint selection based on the validation loss. The Day-2 dataset is reserved for evaluation of different schemes, so as to avoid same-day sample leakage and to assess cross-day generalization; after discarding incomplete sessions it contains $10{,}752$ CPI frames.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.75\linewidth]{figures/scene_schematic.png}
\caption{Geometry of the dual-BS cooperative monostatic measurement scene.}
\Description{Schematic of the sensing plane with BS-0, BS-1, and target sampling locations over the center, side, and corner zones.}
\label{fig:scene}
\end{figure}

\subsection{Experimental Results}
The following performance metrics are evaluated:
\begin{itemize}
\item \textbf{Global mean absolute error (GMAE):} The mean Euclidean positioning error over all samples in the sensing plane.
\item \textbf{Center mean absolute error (CMAE):} The mean Euclidean positioning error restricted to the center zone with $|x|,|y|\le 0.5\,\mathrm{m}$.
\item \textbf{Latency:} The average per-sample wall-clock runtime of the fusion-center algorithm core on a GPU, after a short warm-up whose passes are discarded. For all schemes the timed stage starts from dual-BS ROI range spectra already available at the fusion center (MUSIC/ESPRIT: subspace ranging and geometric intersection; CMTP: CNN forward). Per-BS range-spectrum preprocessing, data loading, and plotting are excluded.
\end{itemize}
Moreover, the following typical localization schemes are selected as baselines:
\begin{itemize}
\item \textbf{MUSIC:} Classical two-stage pipeline that first estimates per-BS ranges via the MUSIC subspace method and then recovers the planar coordinate by dual-station geometric intersection.
\item \textbf{ESPRIT:} Same two-stage pipeline as MUSIC, except that per-BS ranges are estimated by ESPRIT.
\end{itemize}

Fig.~\ref{fig:range-cdf} depicts the per-BS range MAE CDF of MUSIC and ESPRIT at BS-$0$ and BS-$1$. As shown in the figure, the conventional two-stage schemes first decompose planar localization into two independent single-BS ranging problems, so that range estimation errors are directly propagated into the subsequent geometric intersection. In particular, BS-$0$ exhibits a steeper rise in the low-error region (about $65\%$ of samples below $0.2\,\mathrm{m}$), whereas BS-$1$ reaches the $50\%$ CDF only around $0.45$--$0.5\,\mathrm{m}$, indicating a pronounced asymmetry between the two stations. Moreover, after the initial rise, the BS-$0$ curves form a long plateau, and the $90\%$ CDF is reached only near $2.0\,\mathrm{m}$, with the tail extending to about $2.8$--$3.2\,\mathrm{m}$. This long-tail behavior reveals the limited robustness of subspace ranging under adverse CPI / geometry conditions (e.g., peak ambiguity or lock loss), and explains the large outliers later observed in planar positioning.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.85\linewidth]{figures/methods_range_cdf_compare.png}
\caption{Per-BS range MAE CDF of MUSIC and ESPRIT.}
\Description{Four CDF curves comparing range estimation errors at BS-0 and BS-1 for MUSIC and ESPRIT.}
\label{fig:range-cdf}
\end{figure}

Fig.~\ref{fig:pmae-cdf} compares the global positioning MAE CDF of different schemes. Although the three schemes are close in the extremely low-error region (below $0.3\,\mathrm{m}$), the CMTP curve is shifted leftward over the full distribution. Specifically, CMTP achieves a GMAE of $0.579\,\mathrm{m}$, corresponding to reductions of about $38\%$ and $40\%$ relative to MUSIC ($0.937\,\mathrm{m}$) and ESPRIT ($0.966\,\mathrm{m}$), respectively. At the $80\%$ CDF level, the positioning MAE of CMTP is about $0.9\,\mathrm{m}$, whereas MUSIC and ESPRIT require about $1.9\,\mathrm{m}$. Furthermore, the CMTP tail terminates near $1.8\,\mathrm{m}$, while MUSIC and ESPRIT extend to about $3.4\,\mathrm{m}$. This demonstrates that end-to-end joint decoding effectively suppresses the propagation of single-BS ranging failures into the final coordinate estimate.

Fig.~\ref{fig:bim} further illustrates the spatial distribution of positioning MAE over the $x$--$y$ plane. For MUSIC and ESPRIT, the center strip with $|x|\lesssim 0.5\,\mathrm{m}$ remains relatively accurate (mostly green, MAE below $1\,\mathrm{m}$), but the corners and the region with $x>0.6\,\mathrm{m}$ exhibit severe degradation (orange to dark red, MAE above $2.5$--$3\,\mathrm{m}$). In contrast, the CMTP heatmap is dominated by dark green over the entire sensing plane, with only mild light-green / yellow degradation near the edges (mostly below $1.5\,\mathrm{m}$). Quantitatively, CMTP reduces the CMAE from $0.599\,\mathrm{m}$ (MUSIC) and $0.627\,\mathrm{m}$ (ESPRIT) to $0.355\,\mathrm{m}$, i.e., an improvement of about $41\%$ in the center zone. These results verify that, by jointly modeling complementary dual-BS spectral features through DSSE, RAP, and LFRH, CMTP avoids the error accumulation of the ``range-then-intersect'' pipeline and achieves superior accuracy and spatial robustness over classical subspace baselines.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\linewidth]{figures/methods_rmse_cdf_compare_global.png}
\caption{Positioning MAE CDF of Different Schemes.}
\Description{Global positioning MAE CDF curves comparing MUSIC, ESPRIT, and CMTP.}
\label{fig:pmae-cdf}
\end{figure}

\begin{figure*}[htbp]
\centering
\begin{subfigure}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/music_rmse_heatmap.png}
\caption{MUSIC}
\label{fig:heatmap-music}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/esprit_rmse_heatmap.png}
\caption{ESPRIT}
\label{fig:heatmap-esprit}
\end{subfigure}\hfill
\begin{subfigure}[t]{0.32\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/cnn_rmse_heatmap.png}
\caption{CMTP}
\label{fig:heatmap-cnn}
\end{subfigure}
\caption{Positioning MAE heatmaps over the $x$--$y$ plane for different schemes.}
\Description{Three heatmaps of positioning MAE over the sensing plane for MUSIC, ESPRIT, and CMTP.}
\label{fig:bim}
\end{figure*}

In Table~\ref{tab:perf}, the GMAE, CMAE, and fusion-center latency of different schemes are summarized. The CMTP scheme not only attains the lowest positioning errors, but also achieves the lowest latency of $0.574\,\mathrm{ms}$ per sample, which is only about $42\%$ of MUSIC ($1.369\,\mathrm{ms}$) and $28\%$ of ESPRIT ($2.067\,\mathrm{ms}$). Although CMTP introduces a CNN with approximately $5.0\times 10^{5}$ trainable parameters, it avoids the computational burden of subspace spectral search and explicit dual-circle intersection. Consequently, the proposed scheme simultaneously improves localization accuracy and reduces fusion latency. Overall, the experimental results verify the effectiveness of CMTP-CNN for cooperative monostatic ISAC localization in terms of accuracy, spatial robustness, and runtime efficiency.

\begin{table}[htbp]
\centering
\caption{The Performance of Different Schemes.}
\label{tab:perf}
\begin{tabular}{|c|c|c|c|}
\hline
Scheme & GMAE (m) & CMAE (m) & Latency (ms) \\
\hline
MUSIC  & 0.937 & 0.599 & 1.369 \\
\hline
ESPRIT & 0.966 & 0.627 & 2.067 \\
\hline
CMTP   & 0.579 & 0.355 & 0.574 \\
\hline
\end{tabular}
\end{table}
```
