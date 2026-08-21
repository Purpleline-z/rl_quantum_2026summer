# SimCLR identity-safe three-seed Task 3 protocol

## Frozen design

- Seeds: 42, 79, 123
- Encoder: SimCLR only
- Budgets: 10, 25, 50, 75, 100
- Calibration: utility-validation only; outer test is unavailable during Task 3a and selector screening.
- Durable state: Google Drive results plus epoch checkpoints, 30-minute checkpoints, and minute ETA heartbeats.

## Runtime provenance

- Git SHA at protocol generation: `58d59d6c73dd7ea39c2b5438046136a715dc55a0`
- Python: `3.13.15`
- Torch: `2.11.0+cu128`

## Commands

See `COLAB_TASK3_SIMCLR_COMMANDS.md`; rerun any queue command unchanged after interruption.
