#!/usr/bin/env bash
# fix_gpu.sh — align the NVIDIA kernel module with the 580 userspace driver.
#
# Diagnosis (2026-08-28): userspace was upgraded to 580.173 (libnvidia-compute-580)
# but only the 560.35.03 DKMS kernel module is built, so `nvidia-smi` reports
# "Driver/library version mismatch". A plain reboot keeps loading 560 -> still broken.
# Fix: install/build the matching 580 kernel module, verify it built, THEN reboot.
#
# This script does NOT reboot for you (shared lab machine). It stops before the
# reboot and tells you whether it is safe. Run:  bash fix_gpu.sh
set -uo pipefail

KREL="$(uname -r)"
echo "== Running kernel: $KREL"
echo "== Current state ==================================================="
echo "-- Loaded kernel module version:"
cat /proc/driver/nvidia/version 2>/dev/null | head -1 || echo "   (nvidia module not loaded)"
echo "-- nvidia-smi says:"
nvidia-smi 2>&1 | head -3
echo "-- DKMS status:"
dkms status 2>/dev/null | grep -i nvidia || echo "   (no dkms nvidia entries)"
echo "===================================================================="
echo

read -r -p "Install nvidia-driver-580 (builds the matching 580 kernel module)? [y/N] " ans
if [[ "${ans,,}" != "y" ]]; then
    echo "Aborted. No changes made."
    exit 0
fi

echo "== Installing nvidia-driver-580 (requires sudo) ..."
sudo apt update && sudo apt install -y nvidia-driver-580
rc=$?
if [[ $rc -ne 0 ]]; then
    echo "!! apt install failed (rc=$rc). Nothing swapped; GPU still on old module."
    echo "   If you lack admin rights, ask the sysadmin to install nvidia-driver-580."
    exit $rc
fi

echo
echo "== Verifying the 580 module built for the kernel(s) you might boot ..."
# DKMS builds per-kernel and can finish just after apt returns; check the on-disk
# module version for the running kernel AND the newest installed kernel (grub default).
NEWEST_KREL="$(ls -1 /lib/modules 2>/dev/null | sort -V | tail -1)"
ok=1
for k in "$KREL" "$NEWEST_KREL"; do
    ko="$(find /lib/modules/"$k" -name 'nvidia.ko*' 2>/dev/null | head -1)"
    ver="$(modinfo -F version "$ko" 2>/dev/null)"
    echo "   $k -> nvidia module version: ${ver:-<none>}"
    [[ "$ver" == 580.* ]] || ok=0
done
if [[ $ok -eq 1 ]]; then
    echo "OK: 580 kernel module is built for both the running and newest kernel."
    echo
    echo ">>> SAFE TO REBOOT. After reboot, run:  nvidia-smi   (expect 580.173)"
    echo ">>> Then run the profiling harness:      bash profile.sh"
    echo
    echo "Reboot now with:  sudo reboot"
else
    echo "!! WARNING: could not confirm a built 580 module for $KREL."
    echo "   Do NOT reboot yet — you could boot into a GPU-less state."
    echo "   Inspect:  dkms status ; dpkg -l | grep nvidia-dkms"
    exit 1
fi
