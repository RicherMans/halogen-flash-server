{
  description = "halogen-flash-server as a Nix package (docker -> nix port)";

  inputs = {
    # Pinned (not a branch): resolves without the GitHub REST API and matches
    # the nixpkgs revision this fork's systems already use.
    nixpkgs.url = "github:nixos/nixpkgs/3497aa5c9457a9d88d71fa93a4a8368816fbeeba";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      serverRootfs = system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in import ./nix/image.nix {
          fetchers = pkgs.callPackage ./nix/fetchers.nix { };
        };

      runWrapper = system:
        let pkgs = nixpkgs.legacyPackages.${system};
        in pkgs.callPackage ./nix/run.nix {
          rootfs = serverRootfs system;
        };

    in {
      packages = forAllSystems (system: {
        halogen-server = serverRootfs system;
        halogen-run = runWrapper system;
        default = serverRootfs system;
      });

      apps = forAllSystems (system: {
        default = {
          type = "app";
          program = "${runWrapper system}/bin/halogen-server";
        };
      });
    };
}