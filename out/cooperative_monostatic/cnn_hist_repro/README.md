# Historical ckpt protocol re-eval (late_attn_outer3_session)

checkpoint: models/cnn_s2_tune/late_attn_outer3_session/best_model.pth

## no-filter-outliers (full 10752 frames)
- global: 0.5874
- inner:  0.3649
- outer:  0.8672

## filter-outliers (kept 10547 / 10752)
- global: 0.5897
- inner:  0.3650
- outer:  0.8719

Conclusion: historical Global ≈0.5875 holds under unified no-filter protocol.
Gap vs current BEST≈0.595 is real (training recipe), not mainly eval filter.
