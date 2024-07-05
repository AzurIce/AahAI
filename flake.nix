{
  description = "Btraffic";

  nixConfig = {
    extra-substituters = [
      "https://mirrors.ustc.edu.cn/nix-channels/store"
    ];
    trusted-substituters = [
      "https://mirrors.ustc.edu.cn/nix-channels/store"
    ];
  };


  inputs = {
    nixpkgs.url      = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nur.url = github:nix-community/NUR;
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay, nur, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        overlays = [ (import rust-overlay) (nur.overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        pythonPackages = pkgs.python3Packages;
        lib = pkgs.lib;
        stdenv = pkgs.stdenv;
        rust-tools = pkgs.rust-bin.nightly.latest.default.override {
          extensions = [ "rust-src" ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            pkgs.python311
            # pythonPackages.python
            pythonPackages.venvShellHook
          ] ++ [
            rust-tools
          ] ++ (with pkgs.darwin.apple_sdk.frameworks; pkgs.lib.optionals pkgs.stdenv.isDarwin [
          ]);

          packages = [ pkgs.poetry ];
          venvDir = "./.venv";
          postVenvCreation = ''
              unset SOURCE_DATE_EPOCH
              poetry env use .venv/bin/python
              poetry install
          '';
          postShellHook = ''
              unset SOURCE_DATE_EPOCH
              export LD_LIBRARY_PATH=${lib.makeLibraryPath [stdenv.cc.cc]}
              poetry env info
          '';
        };
      }
    );
}
