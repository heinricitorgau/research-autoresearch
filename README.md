# autoresearch

![teaser](progress.png)

這份 README 比較適合把它當成「學習導覽」來讀。

如果你第一次接觸這個 repo，不用先把它想成一個要立刻跑起來的研究系統；更好的方式，是把它看成一份小型案例，學習一個 LLM 實驗專案可以怎麼被整理得夠小、夠清楚、夠容易比較。

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

**需求**：單張 NVIDIA GPU（目前在 H100 上測過）、Python 3.10+、[uv](https://docs.astral.sh/uv/)

這個 repo 用 [`.python-version`](/Users/test/research-autoresearch/.python-version) 指定 `Python 3.10`。如果你有安裝 `pyenv` 或相容工具，進入專案時通常會自動切到這個版本；沒有的話，也建議手動建立 Python 3.10 環境。

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

如果這五步都能成功完成，你就已經有足夠的環境去閱讀 `train.py` 的輸出、理解整個實驗回圈怎麼運作。

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
train.py        — 實驗主體：模型、optimizer、訓練流程
program.md      — 研究 protocol：agent 的工作規則
pyproject.toml  — 專案依賴
uv.lock         — 依賴鎖定檔，確保環境可重現
```

## 如果你的硬體比較小

這份程式碼目前是以單張 NVIDIA GPU 為主要前提。若你只是想把它當成學習材料，在較小機器上觀察概念，可以先考慮：

1. 換成熵更低的資料集，例如 [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean)
2. 降低 `vocab_size`
3. 在 `prepare.py` 降低 `MAX_SEQ_LEN`
4. 在 `prepare.py` 減少 `EVAL_TOKENS`
5. 在 `train.py` 降低 `DEPTH`
6. 使用更簡單的 `WINDOW_PATTERN`
7. 下修 `TOTAL_BATCH_SIZE`

這些調整不只是為了「讓它能跑」，也很適合幫助你理解模型規模、資料分布、序列長度和評估成本之間的關係。

## 延伸閱讀

如果你想看別人怎麼把這個想法搬到其他平台，下面這些 fork 可以當成延伸材料：

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
