# autoresearch

![teaser](progress.png)

這份專案比較適合把它當成一份「學習用的實驗檔案」來讀，而不只是一般的訓練程式碼。它示範的核心觀念不是如何做出一個功能完整的大型框架，而是如何把研究問題縮到夠小，讓人類或 AI agent 都能在清楚的邊界內持續做實驗。

如果你正在學：

- 小型 LLM training loop 怎麼長
- 研究程式要怎麼切出固定環境與可變實驗面
- 為什麼固定時間預算有助於比較實驗
- 怎麼把 agent 納入研究流程

那這個 repo 很值得慢慢讀。

## 這個 repo 在教什麼

`autoresearch` 展示的是一種很極端但很有教育意義的研究設定：

- 把資料準備、evaluation 與固定常數集中起來
- 把大部分研究變因壓縮到單一檔案
- 把每次訓練限制在固定 5 分鐘
- 用同一個指標反覆比較不同想法

換句話說，這裡不是在教你「怎麼打造最完整的訓練系統」，而是在教你「怎麼建立一個容易反覆做實驗的最小研究場」。

這個想法很適合拿來學習，因為你可以很清楚看到：

1. 哪些東西應該固定，這樣實驗才有可比性。
2. 哪些東西應該保持可變，這樣研究才有空間。
3. 為什麼研究流程不只是寫模型，也包含記錄、判斷與丟棄想法。

訓練程式本身來自 [nanochat](https://github.com/karpathy/nanochat) 的單卡簡化版本。如果你想看專案背景，可以再讀這則 [tweet](https://x.com/karpathy/status/2029701092347630069) 和這則 [tweet](https://x.com/karpathy/status/2031135152349524125)。

## 建議怎麼讀

如果你第一次看這個 repo，建議不要一開始就直接跑 agent。比較好的順序是：

1. 先讀 `README.md`，理解整個實驗設計在解什麼問題。
2. 再讀 `prepare.py`，理解哪些環節被故意固定下來。
3. 接著讀 `train.py`，看真正的研究變因都放在哪裡。
4. 最後讀 `program.md`，理解 agent 是怎麼被要求去做實驗的。

這樣讀下來，你會比較容易分辨：

- 哪裡是基礎設施
- 哪裡是實驗操作面
- 哪裡是研究 protocol

## 三個最重要的檔案

這個 repo 很小，真正最值得學的就是下面三個檔案：

- **`prepare.py`**：固定實驗環境。這裡包含資料下載、BPE tokenizer 訓練、dataloader、evaluation，以及一些不希望在每輪實驗中漂移的常數。
- **`train.py`**：主要實驗面。模型結構、optimizer、超參數、batch 設定與訓練流程幾乎都集中在這裡。
- **`program.md`**：給 agent 的研究操作手冊。它定義的不是模型，而是「怎麼做研究」。

這樣的切法很值得學，因為它把「固定條件」和「研究變因」拆得很清楚。

## 這個實驗制度的重點

這個 repo 最重要的設計，不是模型本身，而是比較規則：

- 每次訓練只有 **5 分鐘 wall clock budget**
- 主要比較指標是 **`val_bpb`**
- `val_bpb` 越低越好
- vocab size 改變時，`val_bpb` 仍然能維持可比較性

這表示你學到的不是單純「怎麼讓 loss 下降」，而是更接近研究實務的問題：

- 在固定時間內，哪個設計學得更有效率？
- 某個改動是真的更好，還是只是更花資源？
- 一個想法值得保留，還是應該被丟棄？

如果你對這一類訓練流程還不熟，這份 ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) 可以作為補充背景。

## 快速開始

如果你想先把專案跑起來，最好的心態是把這一步當成「建立學習環境」，而不是立刻開始最佳化。

**需求**：單張 NVIDIA GPU（目前在 H100 上測過）、Python 3.10+、[uv](https://docs.astral.sh/uv/)

這個 repo 目前用 [`.python-version`](/Users/test/research-autoresearch/.python-version) 固定在 `Python 3.10`。如果你有安裝 `pyenv` 或相容工具，進入專案目錄時通常會自動切到這個版本；沒有的話，也建議手動建立 Python 3.10 環境，這樣比較不容易遇到相容性問題。

```bash
# 1. 安裝 uv（如果還沒有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 切到 Python 3.10（若你使用 pyenv）
pyenv install 3.10 -s
pyenv local 3.10

# 3. 安裝依賴
uv sync

# 4. 準備資料與 tokenizer（一次性，約 2 分鐘）
uv run prepare.py

# 5. 跑一次基線訓練（約 5 分鐘）
uv run train.py
```

如果這五步都能成功完成，代表你已經可以開始讀結果、看 log、理解這個 repo 的實驗節奏。

## 你在這裡可以學到的觀察角度

讀這個 repo 時，可以特別注意下面幾件事：

### 1. 固定條件為什麼重要

`prepare.py` 不只是工具檔，它其實在保護整個實驗制度。當資料處理、evaluation 與時間預算都固定時，你比較能相信不同 run 之間的差異是真的來自 `train.py` 的改動。

### 2. 單一實驗面為什麼有價值

把大部分研究變因集中在 `train.py`，會讓你更容易 review diff、回頭比較不同做法，也更適合讓 agent 自主探索。這是一種刻意縮小研究表面的設計。

### 3. 研究不只是修改程式

`program.md` 很值得學，因為它把研究寫成一個 protocol：先建立 baseline，再做單輪假設、執行、讀結果、決定 keep 或 discard。這種流程感比單次靈感更接近真正的研究工作。

### 4. 小而清楚的系統更適合學習

這個 repo 沒有分散式訓練、沒有大型 config 系統、沒有龐大的抽象層。少了很多工程包袱後，你會更容易直接看到模型、訓練 loop 與實驗制度之間的關係。

## 如果你想開始用 agent

當你已經大致理解 repo 的結構後，再把 agent 拉進來會比較有感。這個專案不是要你完全不碰 code，而是讓你把一部分「研究操作」轉移成對 agent 的指令設計。

一個典型做法是讓 agent 先讀 `program.md`，例如：

```text
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

這時候你可以把 `program.md` 理解成一份研究規則文件：你在設計的不是某一層 attention，而是整個 agent 應該如何挑題、如何記錄、何時保留一個想法。

## 專案結構

```text
prepare.py      — 固定實驗環境：資料、tokenizer、evaluation、常數
train.py        — 實驗主體：模型、optimizer、訓練流程
program.md      — 實驗 protocol：agent 的研究規則
pyproject.toml  — 專案依賴
uv.lock         — 依賴鎖定檔，確保環境可重現
```

## 在小型硬體上學習時可以怎麼調

目前這份程式碼以單張 NVIDIA GPU 為前提。若你只是想把它當成學習材料，而手上的機器比較小，可以把下面幾項當成優先調整方向：

1. 換成熵更低的資料集，例如 [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)。
2. 降低 `vocab_size`。
3. 在 `prepare.py` 降低 `MAX_SEQ_LEN`。
4. 在 `prepare.py` 減少 `EVAL_TOKENS`。
5. 在 `train.py` 先從降低 `DEPTH` 開始。
6. 嘗試把 `WINDOW_PATTERN` 改成更簡單的設定。
7. 下修 `TOTAL_BATCH_SIZE`，但盡量維持 2 的冪次。

這些不只是「讓它能跑」，更重要的是幫你觀察模型規模、資料分布、序列長度與評估成本之間的關係。

## 值得參考的 forks

如果你想看別人怎麼把相同概念帶到其他平台，下面這些 fork 可以當延伸閱讀：

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
