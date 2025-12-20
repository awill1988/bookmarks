{
  description = "bookmarks";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        lib = nixpkgs.lib;
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfreePredicate = pkg:
              let
                name = lib.getName pkg;
                hasNvidiaPrefix = lib.hasPrefix "cuda" name
                  || lib.hasPrefix "libcu" name || lib.hasPrefix "libnv" name
                  || lib.hasPrefix "libnpp" name;
              in hasNvidiaPrefix;
          };
        };
        isDarwin = pkgs.stdenv.isDarwin;

        docker-build = pkgs.writeShellScriptBin "docker-build" ''
          set -e
          IMAGE_TAG=''${1:-bookmarks:latest}
          echo "building docker image: $IMAGE_TAG"
          ${pkgs.docker}/bin/docker build -t "$IMAGE_TAG" .
          echo "image built and tagged: $IMAGE_TAG"
        '';

      in {
        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.cargo
            pkgs.rustc
            pkgs.cmake
            pkgs.pkg-config
            pkgs.gcc
            pkgs.openssl
            pkgs.openssl.dev
            pkgs.zlib
            pkgs.docker
            docker-build
          ] ++ (if isDarwin then [ ] else [ pkgs.cudatoolkit ]);

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.openssl.out}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            export PKG_CONFIG_PATH="${pkgs.openssl.dev}/lib/pkgconfig:${pkgs.zlib.dev}/lib/pkgconfig''${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

            # enable wsl nvidia gpu driver libraries when running under wsl
            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi

            # ensure required directories exist
            mkdir -p .cache data
          '';
        };
      });
}
