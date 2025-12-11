{
  description = "bookmarks";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python_pkgs = pkgs.python312Packages;
        python = python_pkgs.python;
      in {
        devShells.default = pkgs.mkShell {
          packages = [
            python
            pkgs.uv
            python_pkgs.hatchling
            pkgs.stdenv.cc.cc
            pkgs.glibc
            pkgs.glibc.dev
          ];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ]}:$LD_LIBRARY_PATH
            export LIBRARY_PATH=${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc pkgs.glibc ]}:$LIBRARY_PATH
            export CPATH=${pkgs.glibc.dev}/include:$CPATH
          '';
        };
      });
}
