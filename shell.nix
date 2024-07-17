let
  pkgs = import <nixpkgs> {};
in pkgs.mkShell {
  packages = [
    (pkgs.python311.withPackages (ps: with ps; [
      (opencv4.override {
        enableGtk3 = true;
        gtk3 = pkgs.gtk3;
      })
      numpy
      matplotlib
      pyqt6
      scipy
    ]))
    pkgs.git
  ];
}
