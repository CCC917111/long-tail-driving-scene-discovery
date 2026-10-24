# Method and historical context

The long method description has moved here from the original README. The
architecture and defaults below describe the method; numerical examples are
not new validated results. The [historical table](../results/README.md) and
[current evaluation protocol](evaluation.md) must be read separately.

The core hypothesis is that a tail-guided sparse representation can improve
long-tail candidate ranking or inspection over cheaper supervised models on
identical frozen VLM features. The controlled comparison is Logistic regression,
MLP, AbsTopK without reward and the same AbsTopK model with reward. A negative
result is useful. Existing legacy entry-point defaults are not single-factor
ablations; see the historical-results caveats.

## Pipeline

Here is an introduction to the pipeline. The figure gives the overview, and
each step is described in its own section below it.

![The six-step pipeline](figures/pipeline.webp)

The pipeline has six steps. **(1)** A single-frame, six-camera driving sample
**(2)** is read by a frozen vision-language model under a scene-description
prompt, and its layer-28 hidden state is averaged over tokens into one vector
*h*. **(3)** A sparse autoencoder encodes *h* into a sparse code split into a
normal part *z<sub>n</sub>* and a long-tail part *z<sub>t</sub>*; during
training, weak frame-level labels suppress *z<sub>t</sub>* on normal samples
and activate it on long-tail ones. **(4)** At inference, the number of active
long-tail units is counted and **(5)** a sample is long-tail as soon as one of
them fires. **(6)** The output is the list of mined samples, the named units
that fired for each, and the evaluation metrics.

### 1. What Counts as Long-Tail

A sample is a long-tail candidate when it is both relatively rare in natural
driving data and likely to require additional defensive driving from the ego
vehicle — slowing down, yielding, keeping a larger safety margin, or paying
closer attention to a nearby risk. The rare-category taxonomy of SearchAD
[[11]](#ref-11) serves as the empirical reference for underrepresented content;
the behavioural criterion links rarity to what matters for driving. The
operational rubric is a strict *normal core*: a frame is normal only if nothing
visible calls for more than routine driving. Its inclusion and exclusion
criteria are in [`latest_grading_criteria.md`](../latest_grading_criteria.md).

The historical capstone report describes large-VLM pre-labelling followed by
manual review. The original labels, review records and prompts must be recovered
before treating those labels as independently verified ground truth. Agreement
between two VLMs does not measure human-label accuracy. Raw-VLM and SAE scores
cannot establish a controlled gain until their test IDs and labels are reconciled
([historical evidence](../results/README.md)).

[`scripts/annotate_normal_core.py`](../scripts/annotate_normal_core.py) applies the
rubric with any Hugging Face VLM and reads the answer from the output-token
logits rather than parsing free text; the prompts are in
[`docs/prompt_templates.md`](prompt_templates.md).

### 2. Input: One Multi-View Keyframe

A sample is one timestamp seen by six synchronised cameras. The code is written
for this format and checks it; the complete input contract is in
[user guide](user.md).

### 3. Hidden-State Extraction

[`scripts/extract.py`](../scripts/extract.py) feeds the six views, in a fixed
order, to Cosmos-Reason1-7B [[14]](#ref-14), a physical-reasoning VLM built on
Qwen2.5-VL [[15]](#ref-15), together with a scene-description prompt that asks
for scene type, weather and lighting, road layout, dynamic objects and hazards.
The model is *not* asked whether the scene is long-tail; the prompt only makes
it build a rich internal description, following the scene-description strategy
of SearchAD [[11]](#ref-11). A forward hook captures the output of the MLP block
(`down_proj`) of decoder layer 28, the last one, and the token states are
averaged:

```math
h_i = \frac{1}{M}\sum_{m=1}^{M} h_{i,m}, \qquad h_i \in \mathbb{R}^{3584},
```

where *M* is the number of tokens and *h<sub>i,m</sub>* the state of token *m*.

### 4. Sparse Decomposition

The SAE encoder maps *h* to a 7,168-unit code. Sparsity follows the k-sparse
autoencoder [[16]](#ref-16) in its Top-K form [[17]](#ref-17) with the absolute
selection rule of AbsTopK [[18]](#ref-18): the k = 512 units with the largest
|activation| are kept with their sign, the rest are set to zero. Keeping large
*negative* activations matters here, because a feature can be expressed by
either sign. The code is split into two halves,
*z* = [*z<sub>n</sub>*, *z<sub>t</sub>*], and the decoder reconstructs *h* from
the whole code. On a mini-batch of *B* samples, with *y<sub>i</sub>* = 1 for a
long-tail sample and 0 for a normal one, the training loss is

```math
\mathcal{L} = \frac{1}{BD}\sum_{i=1}^{B} (1+\alpha y_i) \lVert \hat{h}_i - h_i \rVert_2^2
+ \frac{\beta_{\mathrm{normal}}}{|\mathcal{N}|} \sum_{i \in \mathcal{N}} \lVert z_{t,i} \rVert_2^2
- \frac{\beta_{\mathrm{tail}}}{|\mathcal{T}|} \sum_{i \in \mathcal{T}} \min\left( \lVert z_{t,i} \rVert_2 , \tau \right),
```

where *D* = 3,584 is the feature dimension and $`\mathcal{N}`$ and $`\mathcal{T}`$ are the
normal and the long-tail samples of the batch. The first term reconstructs the hidden state,
weighting the rarer long-tail samples (1 + α) times more. The second keeps the
long-tail subspace silent on normal samples. The third rewards long-tail samples
for activating it, up to the cap τ, beyond which there is nothing more to
gain. These terms encourage a task-sensitive sparse representation. Whether it improves
review yield and whether units support useful explanations are empirical questions
for matched baselines and independent test review; cited SAE work motivates the
design but does not establish either result for this dataset. Defaults: α = 1,
β<sub>normal</sub> = 0.1, β<sub>tail</sub> = 0.5, τ = 2.

### 5. Decision

At inference no label is used. The long-tail units that are active are counted,
and a sample is long-tail as soon as one of them fires:

```math
c_{\mathrm{tail}}(x_i) = \sum_{j=1}^{d_t} \mathbb{1}\left[ \lvert z_{t,i}^{(j)} \rvert > \eta \right],
\qquad
\hat{y}_i = \mathbb{1}\left[ c_{\mathrm{tail}}(x_i) \ge 1 \right],
\qquad \eta = 0.01,
```

where *d<sub>t</sub>* = 3,584 is the number of long-tail units. This is the original fixed rule. The current evaluator additionally selects a
separate score operating point using validation only; both are reported separately.
‖*z<sub>t</sub>*‖<sub>2</sub> is reported alongside as a continuous score, which
orders the flagged samples and is used for the ranking metrics (AUC and AP,
defined under [evaluation protocol](evaluation.md)).

### 6. Output

Each flagged sample comes with the units that fired for it. A sparse code can
be inspected in the [neuron explorer](neurons.md); a semantic name is a human
interpretation to validate, rather than a guaranteed explanation of causation.


### What an SAE Is

A sparse autoencoder (SAE) is a small network trained on the internal
activations of another model. It encodes an activation vector *h* into a much
wider code *z* in which only a few entries are non-zero, and reconstructs *h*
from that code. In this project both maps are single linear layers, so the
reconstruction is a sum of a few decoder columns:

```math
z = \mathrm{AbsTopK}_k\left( W_{\mathrm{enc}} h + b_{\mathrm{enc}} \right),
\qquad
\hat{h} = W_{\mathrm{dec}} z + b_{\mathrm{dec}}
= \sum_{j : z^{(j)} \neq 0} z^{(j)} d_j + b_{\mathrm{dec}},
```

where AbsTopK<sub>k</sub> keeps the *k* entries of largest magnitude and sets
the rest to zero.

This is sparse dictionary learning. Each decoder column *d<sub>j</sub>* is a
*feature direction* in the model's hidden space, and each input is explained
by the handful of directions that are active for it
[[19]](#ref-19), [[20]](#ref-20). Sparsity is what makes the result readable:
because only k = 512 of the 7,168 units may be active for any sample,
information cannot be spread thinly over all of them, and different units are
pushed to specialise in different factors.

The reason for this detour is that a model's own neurons are a poor basis for
interpretation. Individual neurons of large networks are typically
*polysemantic*, responding to several unrelated concepts, because the network
represents more features than it has dimensions [[20]](#ref-20). The cited studies investigate whether SAE directions are more *monosemantic*. In language models they line
up with concepts such as legal text or a particular language, and they are more
interpretable than directions found by other decomposition methods
[[19]](#ref-19), [[20]](#ref-20). The approach scales to production-size models
[[21]](#ref-21) and carries over to vision-language models, where sparse
features make visual representations markedly more monosemantic
[[22]](#ref-22). These studies motivate testing sparse representations for this task. They do
not establish the accuracy, purity or causal meaning of this project's neurons.

## References

1. <a id="ref-1"></a>H. X. Liu and S. Feng. "Curse of rarity" for autonomous
   vehicles. 2022. [arXiv:2207.02749](https://arxiv.org/abs/2207.02749)
2. <a id="ref-2"></a>T. Woodlief, F. Toledo, S. Elbaum and M. B. Dwyer. S3C:
   Spatial semantic scene coverage for autonomous vehicles. *ICSE*, 2024.
   [doi:10.1145/3597503.3639178](https://doi.org/10.1145/3597503.3639178)
3. <a id="ref-3"></a>W. Ding, C. Xu, M. Arief, H. Lin, B. Li and D. Zhao. A
   survey on safety-critical driving scenario generation — a methodological
   perspective. *IEEE T-ITS*, 2023.
   [arXiv:2202.02215](https://arxiv.org/abs/2202.02215)
4. <a id="ref-4"></a>M. Ye, G. P. Meyer, Z. Zhang, D. Park, S. K. Mustikovela,
   Y. Chai and E. M. Wolff. VLMine: Long-tail data mining with vision language
   models. *WACV Workshops*, 2025.
   [arXiv:2409.15486](https://arxiv.org/abs/2409.15486)
5. <a id="ref-5"></a>A. Guillen-Perez. Semantic-Drive: Democratizing long-tail
   data curation via open-vocabulary grounding and neuro-symbolic VLM
   consensus. 2025. [arXiv:2512.12012](https://arxiv.org/abs/2512.12012)
6. <a id="ref-6"></a>D. Bogdoll, N. Ollick, T. Joseph, S. Pavlitska and J. M.
   Zöllner. UMAD: Unsupervised mask-level anomaly detection for autonomous
   driving. 2024. [arXiv:2406.06370](https://arxiv.org/abs/2406.06370)
7. <a id="ref-7"></a>C. M. Jiang, M. Najibi, C. R. Qi, Y. Zhou and D. Anguelov.
   Improving the intra-class long-tail in 3D detection via rare example mining.
   *ECCV*, 2022. [arXiv:2210.08375](https://arxiv.org/abs/2210.08375)
8. <a id="ref-8"></a>D. Feng, X. Wei, L. Rosenbaum, A. Maki and K. Dietmayer.
   Deep active learning for efficient training of a LiDAR 3D object detector.
   *IEEE IV*, 2019. [arXiv:1901.10609](https://arxiv.org/abs/1901.10609)
9. <a id="ref-9"></a>A. Ghita, B. Antoniussen, W. Zimmer, R. Greer, C. Creß,
   A. Møgelmose, M. M. Trivedi and A. C. Knoll. ActiveAnno3D — an active
   learning framework for multi-modal 3D object detection. *IEEE IV*, 2024.
   [arXiv:2402.03235](https://arxiv.org/abs/2402.03235)
10. <a id="ref-10"></a>C. Xu, A. Petiushko, D. Zhao and B. Li. DiffScene:
    Diffusion-based safety-critical scenario generation for autonomous
    vehicles. *AAAI*, 2025.
    [ojs.aaai.org/index.php/AAAI/article/view/32951](https://ojs.aaai.org/index.php/AAAI/article/view/32951)
11. <a id="ref-11"></a>F. Embacher, J. Uhrig, M. Cordts and M. Enzweiler.
    SearchAD: Large-scale rare image retrieval dataset for autonomous driving.
    *CVPR*, 2026. [arXiv:2604.08008](https://arxiv.org/abs/2604.08008)
12. <a id="ref-12"></a>H. Caesar, V. Bankiti, A. H. Lang et al. nuScenes: A
    multimodal dataset for autonomous driving. *CVPR*, 2020.
    [arXiv:1903.11027](https://arxiv.org/abs/1903.11027)
13. <a id="ref-13"></a>Qwen Team. Qwen3.5-397B-A17B. 2026.
    [huggingface.co/Qwen/Qwen3.5-397B-A17B](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
14. <a id="ref-14"></a>NVIDIA. Cosmos-Reason1: From physical common sense to
    embodied reasoning. 2025. [arXiv:2503.15558](https://arxiv.org/abs/2503.15558)
15. <a id="ref-15"></a>S. Bai et al. Qwen2.5-VL technical report. 2025.
    [arXiv:2502.13923](https://arxiv.org/abs/2502.13923)
16. <a id="ref-16"></a>A. Makhzani and B. Frey. k-sparse autoencoders. *ICLR*,
    2014. [arXiv:1312.5663](https://arxiv.org/abs/1312.5663)
17. <a id="ref-17"></a>L. Gao, T. Dupré la Tour, H. Tillman, G. Goh, R. Troll,
    A. Radford, I. Sutskever, J. Leike and J. Wu. Scaling and evaluating sparse
    autoencoders. 2024. [arXiv:2406.04093](https://arxiv.org/abs/2406.04093)
18. <a id="ref-18"></a>X. Zhu, M. M. Khalili and Z. Zhu. AbsTopK: Rethinking
    sparse autoencoders for bidirectional features. *ICLR*, 2026.
    [arXiv:2510.00404](https://arxiv.org/abs/2510.00404)
19. <a id="ref-19"></a>H. Cunningham, A. Ewart, L. Riggs, R. Huben and
    L. Sharkey. Sparse autoencoders find highly interpretable features in
    language models. 2023. [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)
20. <a id="ref-20"></a>T. Bricken, A. Templeton, J. Batson et al. Towards
    monosemanticity: Decomposing language models with dictionary learning.
    *Transformer Circuits Thread*, 2023.
    [transformer-circuits.pub/2023/monosemantic-features](https://transformer-circuits.pub/2023/monosemantic-features)
21. <a id="ref-21"></a>A. Templeton, T. Conerly, J. Marcus et al. Scaling
    monosemanticity: Extracting interpretable features from Claude 3 Sonnet.
    *Transformer Circuits Thread*, 2024.
    [transformer-circuits.pub/2024/scaling-monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
22. <a id="ref-22"></a>M. Pach, S. Karthik, Q. Bouniot, S. Belongie and
    Z. Akata. Sparse autoencoders learn monosemantic features in
    vision-language models. *NeurIPS*, 2025.
    [arXiv:2504.02821](https://arxiv.org/abs/2504.02821)
23. <a id="ref-23"></a>Qwen Team. Qwen3.5-9B. 2026.
    [huggingface.co/Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
24. <a id="ref-24"></a>Qwen Team. Qwen3-VL-2B-Instruct. 2025.
    [huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
25. <a id="ref-25"></a>OpenAI. GPT-4o system card. 2024.
    [arXiv:2410.21276](https://arxiv.org/abs/2410.21276)
26. <a id="ref-26"></a>Google DeepMind. Gemini 3 Pro model card. 2025.
    [deepmind.google/models/model-cards/gemini-3-pro](https://deepmind.google/models/model-cards/gemini-3-pro/)
27. <a id="ref-27"></a>Anthropic. Introducing Claude Haiku 4.5. 2025.
    [anthropic.com/news/claude-haiku-4-5](https://www.anthropic.com/news/claude-haiku-4-5)
28. <a id="ref-28"></a>D. P. Kingma and J. Ba. Adam: a method for stochastic
    optimization. *ICLR*, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

