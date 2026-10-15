23 clips with all 4 configurations

| configuration | mean frames | mean TTFT ms | flow / analysis ms (critical path) | ViT ms | image prefill ms | text prefill ms | other ms |
|---|---|---|---|---|---|---|---|
| GT flow files | 6.09 | 11528 | 331 | 6542 | 3872 | 621 | 162 |
| live OFA, serial | 4.52 | 10548 | 2271 | 4917 | 2826 | 482 | 51 |
| live OFA, pipelined | 4.52 | 8566 | 324 | 4883 | 2821 | 487 | 51 |
| live OFA, pipelined + anchor-first | 4.52 | 8391 | 398 (exposed 0) | 4997 | 2830 | 493 | 71 |

Live flow per clip (mean over clips; serial = each stage timed alone):
  serial decode_sum_ms      765 ms
  serial prep_ms            212 ms
  serial ofa_ms            1236 ms
  serial readback_ms         15 ms
  serial stats_ms            40 ms
  serial wall_ms           2269 ms
  pipelined wall              322 ms  (decode wall 134 ms)
  pipelined + anchor wall     396 ms
  pairs per clip             45.3

Selection, live OFA vs GT: same count 14/23, identical frames 77/104; serial and pipelined pick the same frames on 23/23 clips

Per clip TTFT ms (frames):
| clip | GT flow files | live OFA, serial | live OFA, pipelined | live OFA, pipelined + anchor-first |
|---|---|---|---|---|
| alley_1 | 6279 (3) | 8334 (3) | 6019 (3) | 5689 (3) |
| alley_2 | 5985 (3) | 7888 (3) | 5771 (3) | 5467 (3) |
| ambush_2 | 25301 (14) | 16505 (8) | 14832 (8) | 15003 (8) |
| ambush_4 | 20375 (11) | 15650 (8) | 14268 (8) | 14536 (8) |
| ambush_5 | 19660 (11) | 16307 (8) | 14260 (8) | 13955 (8) |
| ambush_6 | 19404 (11) | 7992 (4) | 7230 (4) | 7120 (4) |
| ambush_7 | 7451 (4) | 9462 (4) | 7419 (4) | 7124 (4) |
| bamboo_1 | 5769 (3) | 7990 (3) | 5726 (3) | 5934 (3) |
| bamboo_2 | 6294 (3) | 7984 (3) | 6021 (3) | 6506 (3) |
| bandage_1 | 5843 (3) | 8031 (3) | 5743 (3) | 5797 (3) |
| bandage_2 | 5774 (3) | 7881 (3) | 5713 (3) | 5470 (3) |
| cave_2 | 19613 (11) | 16319 (8) | 14229 (8) | 13937 (8) |
| cave_4 | 8018 (4) | 10275 (4) | 8175 (4) | 7988 (4) |
| market_2 | 6166 (3) | 8092 (3) | 5787 (3) | 5512 (3) |
| market_5 | 20703 (11) | 17145 (8) | 15446 (8) | 15923 (8) |
| market_6 | 14472 (8) | 9453 (4) | 7639 (4) | 7495 (4) |
| mountain_1 | 5819 (3) | 7844 (3) | 5761 (3) | 5609 (3) |
| shaman_2 | 6400 (3) | 8144 (3) | 5872 (3) | 5806 (3) |
| shaman_3 | 5965 (3) | 8057 (3) | 5979 (3) | 5917 (3) |
| sleeping_1 | 6539 (3) | 7945 (3) | 5791 (3) | 5484 (3) |
| sleeping_2 | 5985 (3) | 7988 (3) | 5750 (3) | 5443 (3) |
| temple_2 | 14360 (8) | 9501 (4) | 7421 (4) | 7133 (4) |
| temple_3 | 22973 (11) | 17808 (8) | 16175 (8) | 14141 (8) |

figure: /home/nvidia/Desktop/Projects/VLM/VLM-Pipeline/dmc955/orin_session/live_flow/results/fig_ttft_timeline.png
