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
- 正常值參考:val_bpb ≈ 1.38–1.40、peak_vram_mb ≈ 6160、tok/sec ≈ 60–67k、attention backend 應顯示 `flex-attention`
- 其他平台(uv、Apple Silicon)見 `README.md`

## 比較規則(補充 program.md)

- seed 固定用預設 42;`AUTORESEARCH_SEED` 環境變數只用於噪音量測
- 同 seed 的 run-to-run 噪音實測約 ±0.017 bpb(時間制預算下步數會浮動)。小於噪音水位的 val_bpb 差異不構成 keep 的理由;最新量測值記錄在 `results.tsv` 開頭幾列
