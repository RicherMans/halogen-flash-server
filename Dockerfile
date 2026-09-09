# deploy/Dockerfile — the halogen-flash-server fork image.
#
# The engine (flash_serve) and the OpenAI front-end (serve_api.py) are closed
# and ship in the public release image, so this is a THIN derivative: it bases
# on that image and layers only this fork's additions on top.
#
# The only behavioural addition is the llama-swap bridge, OFF unless
# HALOGEN_LLAMA_SWAP=1. Unset (the shipped state), behaviour is byte-identical
# to the upstream image.
#
#   docker build -t ghcr.io/<you>/halogen-flash-server:0.5.6 -f Dockerfile .
#
# Usage (llama-swap front-end, bridge exposed):
#   docker run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups \
#     --security-opt seccomp=unconfined --ipc=host --ulimit memlock=-1:-1 \
#     -e HALOGEN_LLAMA_SWAP=1 -p 8732:8732 \
#     -v ~/halogen-models:/models:ro \
#     ghcr.io/<you>/halogen-flash-server:0.5.6
#
# A client's -p maps the outside port to the BRIDGE on 8732; the api on 8731
# and the engine on 8730 stay inside the container, unpublished.

FROM ghcr.io/peonist-ai/halogen-flash-server:0.5.6

COPY deploy/entrypoint.sh /usr/local/bin/entrypoint.sh
COPY deploy/llama-swap-bridge.py /usr/local/bin/llama-swap-bridge.py
RUN chmod +x /usr/local/bin/entrypoint.sh /usr/local/bin/llama-swap-bridge.py

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["all"]