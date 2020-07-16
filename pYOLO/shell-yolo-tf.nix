{ pkgs ? import <nixpkgs> {}, devBuild ? true }:

let
  matplotlib = pkgs.python37.pkgs.matplotlib.override { enableGtk3 = true; };
  python-re3 = pkgs.python37.withPackages (ps: with ps; [
    #opencv3
    #tensorflow
    tensorflowWithCuda
    numpy
    pandas
    h5py
    scipy
    numba
    colorama
    cython
    pillow
    wget
    seaborn
    easydict
  ]);
  opencv3-re3 = pkgs.python37.pkgs.opencv3.override { enableGtk3 = true; enableFfmpeg = true; };

in
  pkgs.mkShell {
   name = "yolo-tf-shell";
   buildInputs = [
      python-re3
      opencv3-re3
      matplotlib

    ];
  }
