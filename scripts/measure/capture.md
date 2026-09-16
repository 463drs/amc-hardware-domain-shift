Bench: HackRF One (clone, r9) -> 30 dB attenuator -> RG-58 cable, 3 m -> RTL-SDR Blog V4 (R828D)
Both devices on USB of the same host. Bias tee off (rtl_biast -b 0).
Date: <fill in>

tone_cap.bin - 20 s, 40 MB
  TX: hackrf_transfer -t tone_300k.bin -f 433000000 -s 2048000 -x 32 -a 0 -R
  RX: rtl_sdr -f 433000000 -s 1024000 -g 19.7 -n 20480000 tone_cap.bin
  Tone at +300 kHz offset (observed 298.3 kHz due to reference oscillator mismatch).
  Level: max |x| = 0.934, no clipping.

noise_cap.bin - 10 s, 20 MB
  HackRF disconnected physically. Attenuator screwed onto the RTL-SDR input, far end open.
  RX: rtl_sdr -f 433000000 -s 1024000 -g 19.7 -n 10240000 noise_cap.bin

Format: unsigned bytes, I and Q interleaved, mid-scale 127.5.