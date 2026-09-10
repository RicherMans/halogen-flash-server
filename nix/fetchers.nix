# nix/fetchers.nix — fetch docker layers/config from a registry and merge them
# into a rootfs the way docker would.
#
# This is a small, self-contained port of awakesecurity/hocker's
# fetchDockerConfig / fetchDockerLayer / fetchdocker (Apache-2.0). The
# packaging differs from hocker in two ways:
#
#   * The fetchers derive an ANONYMOUS ghcr.io bearer token themselves at
#     build time (ghcr requires a Bearer header even for public blobs), so no
#     credential is ever stored.
#   * fetchdocker merges the layers into a runnable rootfs (whiteout-correct)
#     instead of producing a docker-load compositor.
#
# Each fetch is a fixed-output derivation, so nix verifies content by the
# pinned sha256 (and sandboxed network access is allowed for FODs).
{ lib, stdenv, curl, jq, gnutar, cacert }:

let
  ghcrTokenUrl = repository: imageName:
    "https://ghcr.io/token?service=ghcr.io"
    + "&scope=repository:${repository}/${imageName}:pull";

  blobUrl = registry: repository: imageName: digestHex:
    "${registry}/${repository}/${imageName}/blobs/sha256:${digestHex}";

  stripSha = d: builtins.replaceStrings [ "sha256:" ] [ "" ] d;

  # Fetches one blob through an anonymous ghcr token into $out.
  fetchBlob = { registry, repository, imageName, tag, digestHex, accept, sha256, ... }:
    stdenv.mkDerivation {
      name = "halogen-blob-${lib.substring 0 16 digestHex}";
      nativeBuildInputs = [ curl jq cacert ];
      outputHashAlgo = "sha256";
      outputHashMode = "flat";
      outputHash = sha256;
      buildCommand = ''
        export CURL_CA_BUNDLE="${cacert}/etc/ssl/certs/ca-bundle.crt"
        token="''$(${curl}/bin/curl -fsSL '${ghcrTokenUrl repository imageName}' | ${jq}/bin/jq -r .token)"
        ${curl}/bin/curl -fsSL \
          -H "Authorization: Bearer ''$token" \
          -H 'Accept: ${accept}' \
          '${blobUrl registry repository imageName digestHex}' -o "''$out"
        echo "fetched sha256:${digestHex} (${tag}) for ${repository}/${imageName}"
      '';
    };
in {
  # The image config JSON (a single file).
  fetchDockerConfig = attrs@{ registry, repository, imageName, tag, configDigest, sha256 }:
    fetchBlob (attrs // {
      digestHex = stripSha configDigest;
      accept = "application/vnd.docker.container.image.v1+json";
    });

  # One rootfs layer, as a gzip tarball (a single file).
  fetchDockerLayer = attrs@{ registry, repository, imageName, tag, layerDigest, sha256 }:
    fetchBlob (attrs // {
      digestHex = stripSha layerDigest;
      accept = "application/vnd.docker.image.rootfs.diff.tar.gzip";
    });

  # Merge the layers in order into a single store rootfs.
  # Whiteouts are applied per layer against the accumulated fs BEFORE the
  # layer's remaining entries are copied in, which is docker's semantics: a
  # `.wh.<name>` deletes a lower-layer path, `.wh..wh..opq` hides everything a
  # lower layer contributed to its directory.
  fetchdocker = { name, tag, imageConfig, imageLayers, ... }:
    stdenv.mkDerivation {
      pname = name;
      version = tag;
      nativeBuildInputs = [ gnutar ];
      passthru = {
        inherit imageConfig imageLayers tag;
      };
      buildCommand = ''
        stage="''$TMPDIR/merge-stage"
        mkdir -p "''$out" "''$stage"
        for layer in ${lib.concatStringsSep " " imageLayers}; do
          rm -rf "''$stage"/*
          tar -xzf "''$layer" -C "''$stage" --numeric-owner --no-same-owner 2>/dev/null \
            || tar -xf "''$layer" -C "''$stage"
          # opaque whiteout: drop everything a lower layer put in that dir
          find "''$stage" -name '.wh..wh..opq' -print0 | while IFS= read -r -d ''' w; do
            rel="''${w#"''$stage"/}"; parent="''$(dirname "''$rel")"
            if [ -d "''$out/''$parent" ]; then rm -rf "''$out/''$parent"/*; fi
            rm -f "''$w"
          done
          # plain whiteout: remove a lower-layer path
          find "''$stage" -name '.wh.*' -print0 | while IFS= read -r -d ''' w; do
            rel="''${w#"''$stage"/}"; dir="''$(dirname "''$rel")"; base="''$(basename "''$rel")"
            rm -rf "''$out/''$dir/''${base#.wh.}"
            rm -f "''$w"
          done
          cp -ra "''$stage"/. "''$out"/
        done
        echo "merged ${name} ${tag}: ''$(du -sh "''$out" | cut -f1)"
      '';
    };
}