# models/ (owner: Anish)

To deliver:
- `snn.py`: 2 convolutional LIF blocks (16 and 32 channels, 2x pooling each), a recurrent LIF layer (128 neurons) and a sigmoid (x, y) readout; one neural update per frame; state kept across frames and reset between videos
- `train_snn.py`: surrogate-gradient training with backpropagation through time, `common/loss.py`, Adam lr 1e-3, batch 16, up to 30 epochs with early stopping on validation
- state-reset ablation at evaluation (`snn_reset`) and spike counts per frame
- prediction files for seeds 0, 1, 2

Order of work: get it learning on fully visible videos first, then add occlusion.
