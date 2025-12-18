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
        python_pkgs = pkgs.python312Packages;
        python = python_pkgs.python;

        isDarwin = pkgs.stdenv.isDarwin;

      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            pkgs.maturin
            pkgs.cargo
            pkgs.rustc
            pkgs.cmake
            pkgs.pkg-config
            pkgs.gcc
            pkgs.zlib
            pkgs.docker
          ] ++ (if isDarwin then [ ] else [ pkgs.cudatoolkit ]);

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # enable wsl nvidia gpu driver libraries when running under wsl
            if [ -d /usr/lib/wsl/lib ]; then
              export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
            fi
	    export MAKEFLAGS="-j8"
	    ${if isDarwin then ''
              export CMAKE_ARGS="-DGGML_METAL=on"
            '' else ''
              export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_PARALLEL_LEVEL=8"
            ''}
          '';
        };

        apps.docker-build = {
          type = "app";
          program = toString (pkgs.writeShellScript "docker-build" ''
            set -e
            IMAGE_TAG=''${1:-bookmarks:latest}
            echo "building docker image: $IMAGE_TAG"
            ${pkgs.docker}/bin/docker build -t "$IMAGE_TAG" .
            echo "image built and tagged: $IMAGE_TAG"
          '');
        };
      });
}
