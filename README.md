# autoresearch

這份 README 比較適合把它當成「學習導覽」來讀。

如果你第一次接觸這個 repo，不用先把它想成一個要立刻跑起來的研究系統；更好的方式，是把它看成一份小型案例，學習一個 LLM 實驗專案可以怎麼被整理得夠小、夠清楚、夠容易比較。現在這個 repo 也已經調整成可在 Apple Silicon 上執行學習版實驗，不再只限於 NVIDIA CUDA 環境。

## Timeline

- 2026-03：專案以最小化自動研究場的形式出現，核心想法是把 `train.py` 當成主要實驗面，讓 agent 在固定 5 分鐘預算內反覆提出、執行、比較並保留實驗。
- 2026-04-09：README 與 `program.md` 被重寫成更偏學習導覽與研究 protocol 的版本，重點從專案宣言轉向「這個 repo 可以學到什麼」與「實驗流程怎麼被制度化」。
- 2026-04-10：專案進一步調整為可在 Apple Silicon 上執行的學習版。依賴不再綁死 CUDA-only `torch`，`train.py` 與 `prepare.py` 加入 `cuda` / `mps` / `cpu` 裝置偵測與 fallback 路徑，並把 Apple Silicon 的預設模型、batch 與數值穩定性設定調得更保守。
- 2026-04-10：同一天也補上了 `train.py` 輸出判讀說明，讓第一次看到 `val_bpb`、`training_seconds`、`total_seconds`、`peak_vram_mb` 等欄位時，可以直接把 log 當成學習材料來讀。
- 2026-07-20：加入 Windows + NVIDIA 消費級 GPU（RTX 4060 8GB 實測）支援：FlexAttention 實作真正的 sliding window（FA3 只支援 Hopper）、依 VRAM 分級的預設值、`GQA_RATIO` 超參數、torch 2.6 相容性修正。同時把平台基礎設施抽成 `runtime.py`，讓 `train.py` 回到純實驗面。

## 這份專案適合誰

這個 repo 很適合下面幾種讀者：

- 想看小型 LLM training loop 長什麼樣子的人
- 想理解研究專案怎麼拆成固定條件與可變條件的人
- 想學怎麼把「做實驗」變成一個可重複流程的人
- 想理解 AI agent 怎麼被放進研究流程裡的人

如果你剛開始接觸模型訓練，這個 repo 的價值不在於功能很多，而在於它把重要觀念壓得很集中。

## 你會在這裡學到什麼

這個 repo 主要在教四件事：

1. **研究專案怎麼縮小**
   不是所有研究都需要大型框架。很多時候，先把問題縮成一個能快速反覆驗證的最小系統，反而更容易學到東西。

2. **什麼該固定、什麼該改**
   如果資料處理、evaluation 和時間預算一直漂移，你很難知道模型變好是因為想法好，還是因為條件偷偷變了。這個 repo 刻意把固定條件集中起來，把真正的研究變因壓到少數位置。

3. **實驗不是單次靈感，而是比較制度**
   這裡最重要的不是某個特定模型技巧，而是你怎麼建立 baseline、怎麼記錄結果、怎麼決定一個改動該保留還是丟棄。

4. **agent 可以參與研究，但前提是規則清楚**
   這個 repo 不只包含訓練程式，也包含 `program.md` 這種「研究 protocol」。它提醒你，讓 agent 做研究，不只是讓它改 code，而是先定義它怎麼工作。

## 先有一個整體概念

`autoresearch` 的核心想法很簡單：把研究回圈縮到最短。

一輪實驗大致上是這樣：

1. 修改 `train.py`
2. 跑一次固定 5 分鐘的訓練
3. 讀取 `val_bpb`
4. 判斷這次改動要保留還是丟棄

這樣的好處是，研究問題會變得很具體：

- 在相同的時間內，哪個設計學得更有效率？
- 某次改動是真的更好，還是只是多吃了資源？
- 一個新想法值得保留，還是只是在增加複雜度？

這跟一般「一直調參直到看起來比較好」的方式不太一樣。這裡更像是在學如何建立一套乾淨的比較規則。

## 建議閱讀順序

如果你是第一次看這個 repo，推薦照下面順序讀：

1. 先讀這份 `README.md`
   目標是先理解這個 repo 想教什麼，而不是先看細節。

2. 再讀 `prepare.py`
   這個檔案幫你看清楚什麼被當成固定實驗環境。資料、tokenizer、evaluation、常數大多在這裡。

3. 接著讀 `train.py`
   這是最重要的實驗面。模型結構、optimizer、超參數、batch 設定、訓練流程幾乎都集中在這裡。

4. 最後讀 `program.md`
   它不是模型程式，而是 agent 的工作說明。讀完後你會更清楚這個專案怎麼把研究流程 formalize 成一份可執行的規則。

照這個順序讀，會比較容易分清楚：

- 哪些東西是基礎設施
- 哪些東西是研究變因
- 哪些東西是研究流程

## 三個最值得學的檔案

### `prepare.py`

這個檔案可以當成「固定實驗環境」來理解。它大致負責：

- 資料下載
- tokenizer 訓練
- dataloader
- evaluation
- 一些不希望每輪實驗都改來改去的常數

讀它的時候，可以問自己一個問題：為什麼這些內容被放在固定區，而不是交給 agent 一起修改？

### `train.py`

這是主要的研究操作面。它包含：

- 模型架構
- optimizer
- 超參數
- batch 設定
- 訓練 loop

你可以把它想成這個 repo 的「實驗桌面」。大部分真正值得比較的想法，都會發生在這裡。

### `program.md`

這個檔案很適合拿來學「研究流程怎麼被文字化」。它的重點不是模型，而是：

- baseline 要怎麼建立
- 實驗結果怎麼記
- 什麼情況 keep
- 什麼情況 discard
- agent 應該怎麼持續工作

如果你平常只把 README 當專案說明，那這個檔案會很值得特別注意，因為它更像是一份研究操作手冊。

## 這個 repo 最重要的設計觀念

### 1. 固定時間預算

每次訓練都只跑 **5 分鐘 wall clock**。

這個設定很值得學，因為它直接把問題從「誰跑得比較久」變成「誰在同樣時間內學得比較有效率」。

### 2. 固定評估指標

主要指標是 **`val_bpb`**，而且越低越好。

這讓每一輪實驗都有一個一致的比較基準。對學習者來說，這也很重要，因為你可以更容易看懂一個改動到底帶來了什麼效果。

### 3. 縮小可變面積

這個 repo 沒有太多分散式訓練設定、巨大的 config 系統或很厚的抽象層。這不是因為那些東西沒價值，而是因為一旦系統太大，學習者就會更難看見真正影響實驗的因素。

### 4. 研究包含「丟棄」

很多人學做模型時，只注意怎麼產生新改動，卻比較少注意怎麼判斷一個改動不值得留下。這個 repo 特別有教育價值的地方，就是它把 keep / discard 也變成流程的一部分。

## 如何先把專案跑起來

如果你想先把專案跑通，可以把這一段當成「建立學習環境」。

**需求**：Python 3.10、[uv](https://docs.astral.sh/uv/)，以及下面任一種裝置環境：

- NVIDIA GPU：最快，也最接近原始設計目標
- Apple Silicon（MPS）：可跑學習與小規模實驗
- CPU：可作為閱讀與功能驗證用途，但速度最慢

這個 repo 現在會根據裝置自動選擇 `cuda`、`mps` 或 `cpu`。在 NVIDIA GPU 上會優先使用原本的 CUDA 最佳化路徑；在 Apple Silicon 上則會自動退回到 PyTorch 內建 attention 與較保守的 optimizer / precision 設定。
另外，Apple Silicon 與 CPU 也會自動使用更小的預設模型與 batch，目標是先讓整個實驗回圈能跑通，而不是硬追原始 H100 配置。

這個 repo 用 [`.python-version`](/Users/test/research-autoresearch/.python-version) 指定 `Python 3.10`。如果你有安裝 `pyenv` 或相容工具，進入專案時通常會自動切到這個版本；如果沒有安裝 `pyenv`，可以直接跳過下面那兩行，改用你現有的 Python 3.10 環境即可。

```bash
# 1. 安裝 uv（如果還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 切到 Python 3.10（只有安裝 pyenv 才需要）
pyenv install 3.10 -s
pyenv local 3.10

# 3. 安裝依賴
uv sync

# 4. 準備資料與 tokenizer（一次性，約 2 分鐘）
uv run prepare.py

# 5. 跑一次基線訓練（約 5 分鐘）
uv run train.py
```

如果這五步都能成功完成，你就已經有足夠的環境去閱讀 `train.py` 的輸出、理解整個實驗回圈怎麼運作。

如果你看到 `zsh: command not found: pyenv`，代表你的電腦沒有安裝 `pyenv`。這不是錯誤本身，只表示你不能用 `pyenv` 切版本；只要你已經有可用的 Python 3.10，直接從 `uv sync` 開始就可以。

在 Windows + NVIDIA 上也可以不裝 `uv`，直接用系統 Python（3.10+）與 pip：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install rustbpe tiktoken pyarrow requests pandas matplotlib "triton-windows<3.3"
python prepare.py
python train.py
```

其中 `triton-windows` 是讓 `torch.compile` 在 Windows 上生效的關鍵（PyPI 的 `triton` 沒有 Windows wheel）；`runtime.py` 啟動時會自動把 `CUDA_PATH` 指向它自帶的 CUDA 工具鏈，不需要另外安裝 CUDA Toolkit。torch 2.6+ 實測可用，不必嚴格對齊 `pyproject.toml` 鎖定的版本。

如果你是在 Apple Silicon 上執行，速度和可調參空間通常都不會和 NVIDIA GPU 相同。比較合理的期待是：

1. 先把它跑通，理解整個實驗回圈
2. 在較小模型與較小 batch 下做學習型實驗
3. 若要追求原始 repo 的吞吐量，再換到 NVIDIA GPU

## 如何讀 `train.py` 的輸出

第一次看到 `uv run train.py` 的輸出時，很容易只盯著最後一行 `val_bpb`。其實這份 log 裡有三層資訊：

1. **這次跑的是什麼模型**
2. **這次訓練過程穩不穩、快不快**
3. **這次實驗最後值不值得拿來比較**

下面用一個 Apple Silicon 上成功跑完的例子來看：

```text
Vocab size: 8,192
Model config: {'sequence_len': 2048, 'vocab_size': 8192, 'n_layer': 4, 'n_head': 2, 'n_kv_head': 2, 'n_embd': 128, 'window_pattern': 'L'}
Parameter counts:
  wte                     : 1,048,576
  value_embeds            : 2,097,152
  lm_head                 : 1,048,576
  transformer_matrices    : 786,560
  scalars                 : 8
  total                   : 4,980,872
Estimated FLOPs per token: 2.359373e+07
Scaling AdamW LRs by 1/sqrt(128/768) = 1.000000
Time budget: 300s
Gradient accumulation steps: 2
step 00364 (99.7%) | loss: 5.632492 | lrm: 0.01 | dt: 1085ms | tok/sec: 15,094 | mfu: 0.0% | epoch: 1 | remaining: 0s
---
val_bpb:          2.007002
training_seconds: 300.1
total_seconds:    783.5
peak_vram_mb:     66.0
mfu_percent:      0.00
total_tokens_M:   6.0
num_steps:        365
num_params_M:     5.0
depth:            4
```

這段輸出可以這樣理解：

- `Vocab size: 8,192`
  這代表 tokenizer 有 8192 個 token。這是目前實驗的離散化基礎，會影響 embedding 大小與輸出層大小。

- `Model config: ...`
  這是這次實驗實際建立出來的模型設定。從這份輸出可以直接讀到：序列長度是 `2048`、總層數是 `4`、embedding 維度是 `128`、attention pattern 是 `L`。如果你在調 `DEPTH`、`HEAD_DIM`、`WINDOW_PATTERN`，這一行就是第一個確認點。

- `Parameter counts: ... total : 4,980,872`
  這代表目前模型大約是 `5.0M` 參數。對 Apple Silicon 來說，這是一個偏學習用途的小模型設定，目標是先穩定跑完整個實驗回圈，而不是追求大模型吞吐。

- `Estimated FLOPs per token: 2.359373e+07`
  這是每個 token 大致需要多少運算量的估計值。它不是最終成績，但可以幫你理解這次模型大概有多重。

- `Scaling AdamW LRs by 1/sqrt(128/768) = 1.000000`
  這表示 learning rate 縮放係數最後被壓在 `1.0`。對目前的 Apple Silicon 路徑來說，這是一個刻意保守的設定，目的是避免小模型時 learning rate 被放太大，導致一開始就 `FAIL` 或數值爆掉。

- `Time budget: 300s`
  這是固定實驗制度的一部分：真正拿來比較的訓練時間是 `300` 秒，也就是 5 分鐘。

- `Gradient accumulation steps: 2`
  這代表一次 optimizer update 會累積 2 個 micro-batch。這通常是為了在較小裝置上維持總 batch 大小，同時控制單步記憶體壓力。

- `step 00364 ... loss: 5.632492 ... tok/sec: 15,094`
  這是訓練過程中的即時狀態列。
  `loss` 是訓練中的交叉熵，主要用來看訓練有沒有爆掉。
  `tok/sec` 是吞吐量，表示目前每秒大約處理多少 token。
  在這個例子裡，Apple Silicon 大約是 `15k tok/sec`，這是一個「能跑通、可觀察」的學習型速度，不是用來追求極限效能的數字。

- `val_bpb: 2.007002`
  這是最重要的最終指標。`bpb` 是 bits per byte，越低越好。之後不同實驗之間最主要就是比較這個值。

- `training_seconds: 300.1`
  這表示真正計入制度的訓練時間約為 300 秒，符合設計預期。

- `total_seconds: 783.5`
  這表示整個程式從開始到結束總共花了約 783 秒。它比 `training_seconds` 大很多，代表除了訓練本身，Apple Silicon 上還有不少時間花在 setup、prefill、evaluation 與其他 overhead。對 Mac 來說，這是正常的，也提醒你 `training_seconds` 才是更重要的比較基準。

- `peak_vram_mb: 66.0`
  這是裝置記憶體峰值的近似觀察值。對目前 Apple Silicon 路徑來說，這個數字偏小，代表這份學習版配置是很保守的。

- `mfu_percent: 0.00`
  `MFU` 是 model FLOPs utilization。這個指標原本比較適合 CUDA/H100 類型的環境，在 Apple Silicon 路徑上目前沒有代表性，所以看到 `0.00` 是正常的。

- `total_tokens_M: 6.0`
  這表示這次 5 分鐘訓練總共處理了大約 `6.0M` token。

- `num_steps: 365`
  這是總共做了多少次 optimizer step。之後如果你調大 batch 或改變吞吐，這個數字也會跟著變。

- `num_params_M: 5.0`、`depth: 4`
  這兩個是最後的摘要資訊，方便你在記錄表裡快速知道這次實驗的模型規模。

對這次輸出，一句話總結就是：**這是一個在 Apple Silicon 上成功跑完整個 5 分鐘訓練回圈的基線學習版實驗**。它的價值不在於 `2.007002` 這個數字本身有多強，而在於它提供了一個穩定、可重複、可比較的起點。之後你若要做實驗，應該拿它當 baseline 去比較，而不是拿它和 H100 上的大模型結果直接相比。

## 如果你要把它當教材來看

下面這幾個觀察角度通常很有幫助：

- 觀察 `prepare.py` 和 `train.py` 的分工
- 觀察哪些設定是固定的，哪些是故意讓實驗去改的
- 觀察 `program.md` 怎麼把研究流程變成規則
- 觀察這個 repo 怎麼避免把研究問題擴散成太大的工程問題

這樣讀，你會比較容易把它內化成方法，而不是只把它當成一份可以執行的 codebase。

## 如果你之後想加入 agent

等你先理解這個 repo 在做什麼之後，再引入 agent 會比較有意義。

一個典型做法是讓 agent 先讀 `program.md`，例如：

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

這時候最值得學的點不是 prompt 本身，而是這件事背後的設計思想：你不是只叫 agent 幫你改一段 code，而是在定義一套 agent 應該如何做研究的規則。

## 專案結構

```text
prepare.py      — 固定實驗環境：資料、tokenizer、evaluation、常數
runtime.py      — 固定平台層：裝置偵測、attention 後端、compile 探測、VRAM 分級
train.py        — 實驗主體：模型、optimizer、訓練流程
program.md      — 研究 protocol：agent 的工作規則
pyproject.toml  — 專案依賴
```

`prepare.py` 和 `runtime.py` 都屬於「不隨實驗改動」的固定區；實驗只發生在 `train.py`。這個切分的目的，是讓實驗 diff 永遠只包含研究變因，而不會混進平台管線的修改。

第一次在新機器上執行 `uv sync` 時，`uv` 會依照目前平台重新解析並建立對應的環境；像 Apple Silicon 和 NVIDIA GPU 這種平台差異較大的情況，這比強行共用同一份舊 lockfile 更安全。

## 如果你的硬體比較小

這份程式碼雖然現在可以在 Apple Silicon 上跑，但原始設計仍然偏向單張 NVIDIA GPU。若你只是想把它當成學習材料，在較小機器上觀察概念，可以先考慮：

1. 換成熵更低的資料集，例如 [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
2. 降低 `vocab_size`
3. 在 `prepare.py` 降低 `MAX_SEQ_LEN`
4. 在 `prepare.py` 減少 `EVAL_TOKENS`
5. 在 `train.py` 降低 `DEPTH`
6. 使用更簡單的 `WINDOW_PATTERN`
7. 下修 `TOTAL_BATCH_SIZE`

這些調整不只是為了「讓它能跑」，也很適合幫助你理解模型規模、資料分布、序列長度和評估成本之間的關係。

## License

MIT
