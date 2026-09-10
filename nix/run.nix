# nix/run.nix — reproduce the container without docker.
#
# Lays the store rootfs out at / (bubblewrap, unprivileged), mounts /proc and
# /dev, binds the weights directory at /models (read-only, like the container's
# :ro), injects the exact ENV defaults the release image bakes, raises the
# memlock limit the way --ulimit memlock=-1 does, then execs the image's own
# /usr/local/bin/entrypoint.sh all — engine on 8730, api on 8731, llama-swap
# bridge on 8732 (on by default in the fork).
#
# GPU devices are dev-bound when present; no seccomp profile is applied, which
# mirrors the image's --security-opt seccomp=unconfined. The network namespace
# is the host's, so the ports are reachable exactly as with -p on a published
# port.
{ lib, pkgs, rootfs }:

let
  # ENV as baked into the release image (docker image inspect .Config.Env).
  bakedEnv = [
    { name = "PATH"; value = "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"; }
    { name = "LANG"; value = "C.UTF-8"; }
    { name = "PYTHONUNBUFFERED"; value = "1"; }
    { name = "HF_HUB_OFFLINE"; value = "1"; }
    { name = "TRANSFORMERS_VERBOSITY"; value = "error"; }
    { name = "HALOGEN_ARCH"; value = "gfx1151"; }
    { name = "HIP_FORCE_DEV_KERNARG"; value = "1"; }
    { name = "HIP_PATH"; value = "/usr/local/lib/python3.12/site-packages/_rocm_sdk_core"; }
    { name = "HIP_DEVICE_LIB_PATH"; value = "/usr/local/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/amdgcn/bitcode"; }
    { name = "ROCM_SITE"; value = "/usr/local/lib/python3.12/site-packages"; }
    { name = "ROCM_PATH"; value = "/usr/local/lib/python3.12/site-packages/_rocm_sdk_core"; }
    { name = "LD_LIBRARY_PATH"; value = "/usr/local/lib/python3.12/site-packages/_rocm_sdk_core/lib:/usr/local/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"; }
    { name = "HALOGEN_ATTN_FA"; value = "64"; }
    { name = "HALOGEN_PROMPT_CACHE"; value = "2"; }
    { name = "HALOGEN_PREFILL_CHUNK"; value = "32768"; }
    { name = "HALOGEN_KV_SLOTS"; value = "4"; }
    { name = "HALOGEN_CTX"; value = "262144"; }
    { name = "HALOGEN_MAX_TOK"; value = "32768"; }
    { name = "HALOGEN_MATMUL_ALGOS"; value = "1"; }
    { name = "HALOGEN_MATMUL_BUCKET"; value = "1"; }
    { name = "HALOGEN_MATMUL_TUNING_FILE"; value = "/opt/halogen/flash-tune.plan"; }
    { name = "HALOGEN_CHECKPOINT"; value = "/models/qwen38-flash-next-w4b.hgn"; }
    { name = "HALOGEN_TOKENIZER"; value = "/tokenizer"; }
    { name = "HALOGEN_PORT"; value = "8730"; }
    { name = "HALOGEN_BIND"; value = "127.0.0.1"; }
    { name = "HALOGEN_API_PORT"; value = "8731"; }
    { name = "HALOGEN_MAX_TOKENS_CAP"; value = "65536"; }
    { name = "HALOGEN_QUEUE_TIMEOUT"; value = "3600"; }
  ];

  setDefaults = lib.concatMapStringsSep "\n"
    (e: ''    export ${e.name}="${e.value}"'')
    bakedEnv;

  gpuBinds = ''
    if [ -e /dev/kfd ]; then ARGS+=(--dev-bind /dev/kfd /dev/kfd); fi
    if [ -e /dev/dri ]; then ARGS+=(--dev-bind /dev/dri /dev/dri); fi
  '';
in
pkgs.writeScriptBin "halogen-server" ''
  #!${pkgs.bash}/bin/bash
  set -euo pipefail

  ROOTFS="${rootfs}"
  unset LC_ALL 2>/dev/null || true

  # Replicate the image's baked ENV: exported unconditionally, exactly as the
  # image config sets it. Caller-set values for a var also named here are
  # replaced, which is what docker -e does against image ENV.
  ${setDefaults}

  # The weights directory. The release image expects them at /models, mounted
  # read-only; HALOGEN_MODELS points the wrapper at them from the host.
  MODES_DIR="''${HALOGEN_MODELS:-/models}"

  ARGS=(
    ${pkgs.bubblewrap}/bin/bwrap
    --die-with-parent
    --bind "$ROOTFS" /
    --proc /proc
    --dev /dev
  )
  ${gpuBinds}
  if [ -d "$MODES_DIR" ]; then
    ARGS+=(--ro-bind "$MODES_DIR" /models)
  fi

  # --ulimit memlock=-1 in the container manifest, same thing here. Silenced:
  # only a privileged caller can raise it, and the container hides the same.
  ulimit -l unlimited 2>/dev/null || true

  exec "''${ARGS[@]}" /usr/local/bin/entrypoint.sh all
''