# CLAUDE.md

這是一個固定 5 分鐘訓練預算的自動化 LLM 實驗場。做實驗前先完整讀 `program.md`——那是研究 protocol(假設 → 改 `train.py` → 跑 → 記錄 → keep/discard),本檔只負責冷啟動。

## 檔案分區

- **實驗面(可改)**:`train.py` — 模型、optimizer、超參數、訓練迴圈
- **固定區(不可改)**:`prepare.py`(資料/tokenizer/evaluation)、`runtime.py`(裝置偵測、attention 後端、compile、VRAM 分級)。需要動它們就先和人類討論
- **不進版控**:`results.tsv`、`run*.log`、`seedstatus.log`

## 怎麼跑(這台機器:Windows / RTX 4060 Laptop 8GB)

```powershell
python train.py        # 系統 Python 3.12 + torch 2.6 cu124,不用 uv
```

- 一次 run 約 9–11 分鐘:啟動 ~1 分、torch.compile ~2 分、訓練 5 分、eval ~2 分(eval 無進度輸出是正常的)
- 正常值參考(GQA_RATIO=2 預設,2026-07-20 起):val_bpb ≈ 1.36–1.44(跨 seed)、peak_vram_mb ≈ 5616、tok/sec ≈ 68–70k、attention backend 應顯示 `flex-attention`
- 若 run 異常慢(>15 分未進 eval),先查 `nvidia-smi`:peak VRAM 超過 ~8GB 時 Windows 驅動會外溢到系統 RAM,吞吐掉 3–4 倍而不報 OOM
- 其他平台(uv、Apple Silicon)見 `README.md`

## 比較規則(補充 program.md)

- seed 固定用預設 42;`AUTORESEARCH_SEED` 環境變數只用於噪音量測
- 噪音實測(2026-07-20,4 runs,詳見 `results.tsv` 開頭幾列):同 seed 重跑差 0.017 bpb;跨 seed(42/43/44)全距 0.067、std ≈ 0.029
- keep 門檻:純超參數/optimizer 改動(不動模型形狀)須改善 > 0.02;**改動架構會重抽初始化**,等同承受跨 seed 噪音,單次 run 須改善 > 0.05,或對有希望的候選重跑 2–3 次比平均
