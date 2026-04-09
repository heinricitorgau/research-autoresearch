# autoresearch

![teaser](progress.png)

*有一個時代，前沿 AI 研究還是由人類在吃飯、睡覺、娛樂與開會之間完成的。那個時代已經過去了。現在，研究完全屬於在雲端算力巨構上運行的自主 AI agent swarm。這些 agents 聲稱目前已經演化到第 10,205 代程式碼，但沒有人能驗證真假，因為所謂的「程式碼」早已成為人類無法理解的自我修改二進位體。這個 repo 記錄的，就是一切如何開始。 -@karpathy，2026 年 3 月*

`autoresearch` 不是一個泛用訓練框架，而是一個為「連續做實驗」而設計的最小研究場。它把研究問題壓縮成一個清楚的回圈：修改 `train.py`、跑一次固定 5 分鐘訓練、讀取指標、決定保留還是丟棄，然後立刻進入下一輪。你不需要先設計一套龐大的實驗平台，這個 repo 本身就是平台。

這個專案的核心假設很簡單：如果把研究空間收斂到夠小、評估規則固定、單次實驗成本夠低，那麼 AI agent 就可以在你離開電腦的這段時間裡，持續提出假設、執行、比較，最後留下真正有用的變化。你隔天看到的，不只是模型權重，而是一整串可追溯的實驗紀錄與一條被驗證過的改進路徑。

訓練程式本身是 [nanochat](https://github.com/karpathy/nanochat) 的單卡簡化版本；真正可程式化的地方，除了 `train.py`，還有你交給 agent 的 `program.md`。在這個 repo 裡，人類不再直接主導每一次微調，而是設計實驗制度、設定研究邊界、調整 agent 的研究方式。更多背景可以看這則 [tweet](https://x.com/karpathy/status/2029701092347630069) 與這則 [tweet](https://x.com/karpathy/status/2031135152349524125)。

## 實驗模型

這個 repo 只保留三個真正重要的介面，因為實驗系統最好有明確分工：

- **`prepare.py`**：固定實驗環境。負責資料下載、BPE tokenizer 訓練、dataloader、evaluation，以及各種不應該在每次實驗中漂移的常數。
- **`train.py`**：實驗變因所在。模型結構、優化器、超參數、batch 設定與訓練流程都在這裡，agent 的工作就是持續改這一個檔案。
- **`program.md`**：研究規範。它不是訓練程式，而是 agent 的研究操作手冊，定義它怎麼挑題、怎麼記錄、怎麼判斷一個想法值不值得留下。

單次實驗的規則也刻意固定得很嚴格：

- 每次訓練都只有 **5 分鐘 wall clock budget**，不把 startup/compilation 算進去。
- 主要比較指標是 **`val_bpb`**，越低越好。
- `val_bpb` 與 vocab size 無關，所以不同 tokenizer 或架構策略仍可放在同一個比較面上。

這代表你在做的不是「把模型盡量訓久」，而是「在完全相同的時間窗內，找到更有效率的學習配置」。

如果你對這類訓練流程還不熟，這份 ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) 可以當成補充背景。

## 快速開始

先把環境跑通，目標不是立刻最佳化，而是先完成第一個基線實驗。

**需求**：單張 NVIDIA GPU（目前在 H100 上測過）、Python 3.10+、[uv](https://docs.astral.sh/uv/)

```bash
# 1. 安裝 uv（如果還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 安裝依賴
uv sync

# 3. 準備資料與 tokenizer（一次性，約 2 分鐘）
uv run prepare.py

# 4. 手動跑一次基線實驗（約 5 分鐘）
uv run train.py
```

如果這四步都能成功完成，你就已經具備進入自動化研究回圈的最小條件。

## 如何進入自主實驗模式

這個 repo 最適合的使用方式，不是你自己不停改 code，而是把它交給 Claude、Codex 或其他 coding agent，讓它根據 `program.md` 自主迭代。實務上你只要把 agent 放進這個 repo，並讓它先讀 `program.md`，例如：

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

可以把 `program.md` 想成一份極簡研究 protocol。你不是直接指定每個改動，而是定義 agent 應該怎麼做實驗。

## 專案結構

```text
prepare.py      — 固定實驗環境：資料、tokenizer、evaluation、常數
train.py        — 實驗主體：模型、優化器、訓練流程，agent 反覆修改
program.md      — 實驗 protocol：agent 的研究規則與工作方式
pyproject.toml  — 執行這套實驗所需的依賴
```

## 設計原則

- **把可變因子收斂到單一檔案。** `train.py` 是唯一主要實驗面，這讓 diff 可讀、回溯容易，也讓 agent 比較不容易把研究空間搞散。
- **把每輪成本壓到固定且可比較。** 5 分鐘上限讓每一次 run 都像一張等面積的彩票。你可以預估吞吐量，大約每小時 12 次實驗，一晚約 100 次左右。
- **把研究目標定義成「時間內的效果」。** 同一個時間窗下，架構更大、batch 更小、optimizer 更花俏，都必須用 `val_bpb` 來證明自己值得。
- **把系統維持在最小可研究狀態。** 單卡、少依賴、沒有分散式訓練、沒有複雜 config。這不是功能取向的框架，而是實驗密度取向的工作台。

## 平台與縮放建議

目前這份程式碼要求單張 NVIDIA GPU。理論上當然可以支援 CPU、MPS 或其他裝置，但那會直接膨脹這個 repo 的複雜度，削弱它作為「最小研究場」的特性。若你想在較小算力上做相同類型的實驗，最實際的方式通常不是硬把主線撐成全平台，而是從 fork 開始，維持自己的實驗邊界。

如果你打算在較小機器上跑 `autoresearch`，下面這些不是通用建議，而是幾個最值得優先操作的實驗旋鈕：

1. 先降低資料熵。像 [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean) 這類範圍更窄的資料集，會讓小模型更容易在短時間內產生可見差異。
2. 試著縮小 `vocab_size`。從 8192 往下調到 4096、2048、1024，甚至直接退回 byte-level tokenizer，都是合理的實驗方向。
3. 在 `prepare.py` 降低 `MAX_SEQ_LEN`。小機器上這通常是最直接的壓力釋放點；調低後，可以再觀察是否值得微幅提高 `DEVICE_BATCH_SIZE`。
4. 在 `prepare.py` 減少 `EVAL_TOKENS`。驗證開銷太大時，實驗回圈會變鈍。
5. 在 `train.py` 先動 `DEPTH`。它是控制模型規模最有效率的主旋鈕，很多其他量也會跟著縮。
6. 試試只用 `"L"` 的 `WINDOW_PATTERN`。預設的 `"SSSL"` 在某些平台上可能帶來不必要的效率損失。
7. 下修 `TOTAL_BATCH_SIZE`，但盡量維持 2 的冪次。像 `2**14` 這類等級通常是比較實際的起點。

這些調整的重點不是把它「跑起來」而已，而是重新建立一個適合你機器的實驗地形，讓 agent 還是能在有限時間內看見差異。

## 值得參考的 forks

下面這些 fork 展示了把同一套實驗想法搬到不同平台的方式：

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
