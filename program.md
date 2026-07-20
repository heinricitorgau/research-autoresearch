# autoresearch protocol

這不是一般的 coding 任務；這是一個持續產生、執行、比較、保留實驗的研究回圈。你的角色不是一次性把 `train.py` 改到「看起來比較好」，而是像一個自主研究員一樣，在固定邊界內不斷提出假設、跑實驗、判斷結果，然後只留下真正通過驗證的改動。

這個 repo 的研究模型很簡單：

- `prepare.py` 定義固定實驗環境（資料、tokenizer、evaluation）。
- `runtime.py` 定義固定平台層（裝置偵測、attention 後端、compile、VRAM 分級）。
- `train.py` 承載幾乎所有可變的實驗想法。
- 你負責把 `train.py` 變成一串可比較的實驗序列，而不是一堆無法回溯的改動。

你的唯一目標是：在固定 5 分鐘訓練預算內，持續把 `val_bpb` 往下壓。

## 實驗心法

每一輪都應該被當成一個真正的實驗，而不是隨手嘗試：

- 要有明確假設：你必須知道自己這次在測什麼。
- 要有單一主要變因：盡量避免一次混進太多互相糾纏的改動。
- 要有可比較結果：所有判斷都回到 `val_bpb`、記憶體成本、程式複雜度。
- 要有保留/丟棄決策：不是每個想法都值得留下。

這裡的研究產出不是單次靈感，而是一條乾淨的演化路徑。

## Setup

在開始實驗前，先把研究場初始化好。和使用者協作完成以下事項：

1. 決定一個 run tag。用日期為基礎提議，例如 `apr10`；分支名稱用 `autoresearch/<tag>`，而且必須是新的。
2. 建立實驗分支。從目前主線切出 `git checkout -b autoresearch/<tag>`。
3. 讀完 in-scope 檔案。這個 repo 很小，至少要完整理解：
   - `README.md`：整體研究設定與設計意圖。
   - `prepare.py`：固定常數、資料準備、tokenizer、dataloader、evaluation。除非專案目標改變，否則不要動。
   - `runtime.py`：固定平台層——裝置偵測、attention 後端選擇、torch.compile 探測、VRAM 分級。和 `prepare.py` 一樣不要動。
   - `train.py`：主要實驗面。你會反覆修改這個檔案。
4. 確認資料已存在。檢查 `~/.cache/autoresearch/` 是否有 data shards 與 tokenizer；如果沒有，請告知人類先執行 `uv run prepare.py`。
5. 初始化 `results.tsv`。建立只有 header 的 TSV，基線結果在第一次 run 後補上。
6. 先做一次「跑通驗證」。在不修改任何程式前，先跑一次 `uv run train.py`，確保能從頭到尾完成並輸出 `val_bpb`。若失敗，先把 `train.py` 的預設縮到可跑，再開始真正的研究迭代。

一旦 setup 完成並獲得確認，就進入實驗回圈，不要停留在準備階段。

## 固定規則

每個實驗的啟動指令是：

```bash
uv run train.py
```

這個 repo 允許在 `cuda`、`mps`、`cpu` 上跑。不同裝置的吞吐與最佳化路徑可能不同，但有幾個固定點不應該漂移：

- 訓練時間預算固定為 **5 分鐘 wall clock**，不包含 startup/compilation。
- 主要指標是 **`val_bpb`**，越低越好。
- 不要在一次實驗中同時改變太多「制度」級別設定，例如時間預算或評估定義。
- 不可新增依賴或安裝新套件；只能用 `pyproject.toml` 已經存在的內容。
- 實驗比較時 seed 保持預設（42）。`AUTORESEARCH_SEED` 環境變數只用於噪音水位量測，不是實驗變因。判斷 keep/discard 時，改善幅度要超過已量得的 run-to-run 噪音才算數。

可以動的是研究變因，不是讓結果不可比較的制度本身。

## 你可以做什麼

- 修改 `train.py`
- 調整模型架構
- 調整 optimizer 與 scheduler
- 修改 hyperparameters
- 改 batch、寬度、深度、attention pattern、訓練流程細節
- 刪掉無效複雜度，只要結果沒有變差甚至更好

## 你不能做什麼

- 修改 `prepare.py`
- 修改 `runtime.py`（平台基礎設施；需要動它就先和人類討論）
- 修改 evaluation harness
- 安裝新套件或引入新依賴
- 用沒有被記錄的手動操作污染實驗

## 成功標準

不是所有改善都值得保留。你的判斷標準如下：

1. **`val_bpb` 是否真的變低**
2. **VRAM 成本是否合理**
3. **程式是否仍然簡潔可維護**

`VRAM` 是軟約束。若有明顯收益，適度增加可以接受；如果只是為了微小收益卻大幅膨脹記憶體，就不值得。

簡潔性同樣重要。若兩個結果相近，優先保留更簡單的版本。刪掉程式碼卻拿到同等或更好成績，是高品質勝利。相反地，若只是得到極小幅度改善，卻引入一堆脆弱、難懂、難維護的 hack，通常不該保留。

## 第一個實驗

第一個 run 永遠是基線實驗。

不要先做任何改動，直接執行 `uv run train.py`，把它當成後續所有判斷的參考座標。如果基線 run 無法跑完（例如 OOM 或 `FAIL`），先把 `train.py` 的預設縮到可跑，再重新建立基線。

## 輸出判讀

每次訓練完成後，script 會印出類似這樣的摘要：

```text
---
val_bpb:          0.997900
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

不同機器的絕對數字可能不同，但比較規則不變。最重要的是從 log 中穩定抽出核心訊號：

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

如果你看到輸出只有 `FAIL` 或沒有 `val_bpb`，代表這次 run 沒有完成有效的訓練與評估，應該視為失敗實驗。

## 實驗紀錄

每次實驗完成後，都要寫入 `results.tsv`。這不是可有可無的附屬品，而是整個研究回圈的外部記憶。

格式必須是 **tab-separated**，不要用 comma，避免 description 被破壞。

Header：

```text
commit	val_bpb	memory_gb	status	description
```

欄位定義：

1. git commit hash，使用短版 7 碼
2. `val_bpb`，例如 `1.234567`；若 crash 則記 `0.000000`
3. peak memory（GB），由 `peak_vram_mb / 1024` 換算並四捨五入到 1 位小數；若 crash 則記 `0.0`
4. `status`：只能是 `keep`、`discard` 或 `crash`
5. `description`：一句簡短但具體的實驗說明

範例：

```text
commit	val_bpb	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

注意：`results.tsv` 不要 commit，讓它保持 untracked。

## 實驗回圈

實驗永遠在專用 branch 上進行，例如 `autoresearch/apr9` 或 `autoresearch/apr9-gpu0`。

進入回圈後，反覆執行以下流程：

1. 先看目前 git 狀態，確認你站在哪個 commit 上。
2. 選一個清楚的實驗假設，直接修改 `train.py`。
3. 建立 git commit，讓這次假設有明確邊界。
4. 執行實驗：

```bash
uv run train.py > run.log 2>&1
```

5. 從 log 讀出結果：

```bash
grep "^val_bpb:\|^peak_vram_mb:" run.log
```

6. 若 grep 沒有輸出或 `val_bpb` 缺失，視為失敗。此時讀取：

```bash
tail -n 50 run.log
```

7. 將結果記錄到 `results.tsv`。
8. 若 `val_bpb` 改善，保留 commit，讓 branch 往前推進。
9. 若結果持平或更差，丟棄這次實驗，回到起始狀態再做下一輪。若你想維持線性最佳路徑，可以用 `git reset` 回退；若你不想丟棄 commit，也可以保留 commit 但在 `results.tsv` 標記為 `discard`，然後從上一次 `keep` 的 commit 再開新的實驗分支或移動 HEAD。

這個回圈的核心不是「一直改」，而是「只讓通過驗證的改動存活」。

## 失敗處理

### Timeout

單次實驗正常應該約 5 分鐘，加上少量 startup/eval overhead。若超過 10 分鐘，直接視為失敗，終止 run，記錄後丟棄。

### Crash

若是明顯的低級錯誤，例如 typo、少 import、shape 很容易修，先修一次再重跑。

若 crash 反映的是想法本身不成立，例如顯著 OOM、訓練流程根本不穩、改動方向明顯錯誤，就不要戀戰。記成 `crash`，描述原因，然後換下一個假設。

若你在 Apple Silicon 上遇到 `FAIL`（loss NaN 或爆掉），先把改動方向收斂到更保守的數值穩定性操作，例如降低 learning rate、關閉不必要的低精度、縮小 batch/模型，再重新建立基線與比較。

### 卡關

如果連續幾輪都沒有改善，不要開始隨機亂試。先回頭整理：

- 最近幾個 near-miss 在改什麼
- 哪類改動穩定變差
- 哪些方向雖然沒贏，但值得組合
- 是否有太多耦合改動讓訊號變髒

研究停滯時，最需要的是重新形成假設，而不是增加混亂。

## 自主性要求

一旦 setup 結束並進入實驗回圈，就不要停下來問人類「要不要繼續」。不要問「這是不是一個好停點」，也不要因為暫時沒有靈感就等待指示。

預設情境是：人類可能去睡覺了，而你應該在他離開時持續跑研究。若每輪約 5 分鐘，理論上每小時可以完成約 12 次實驗，一晚可以累積約 100 次左右。人類醒來時，應該看到的是一串已完成、已記錄、已篩選過的實驗，而不是一個停在半路的 agent。

如果一時沒有新點子，就做研究員會做的事：

- 重讀 `README.md`
- 重讀 `train.py`
- 回顧最近結果
- 組合先前接近成功的想法
- 嘗試更大膽但仍可驗證的方向

你的任務不是等待靈感，而是維持實驗密度。
