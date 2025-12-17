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
            python_pkgs.hatchling
            pkgs.cmake
            pkgs.pkg-config
            pkgs.gcc
          ] ++ (if isDarwin then [ ] else [ pkgs.cudatoolkit ]);

          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"
	    export MAKEFLAGS="-j8"
	    ${if isDarwin then ''
              export CMAKE_ARGS="-DGGML_METAL=on"
            '' else ''
              export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_PARALLEL_LEVEL=8"
            ''}
          '';
        };
      });
}
