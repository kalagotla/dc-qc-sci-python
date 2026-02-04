## LES Cylinder CFD Subset for Hour 2

This folder is intended to hold a **small subset** of the LES cylinder case used in Hour 2, Part 2 of the workshop.

To keep the repository size manageable, only a few files should be checked into GitHub. A good minimal subset is:

- `cylinder.sp.x` – grid file for the cylinder case  
- A handful of flow snapshots, for example:
  - `sol-0000010.q`
  - `sol-0000020.q`
  - `sol-0000030.q`
  - `sol-0000040.q`
  - `sol-0000050.q`

During development, you can copy these from your full LES dataset, e.g. from your `cylinder_data` folder:

- `C:/.../project-arrakis/data/cylinder_data/cylinder.sp.x`
- `C:/.../project-arrakis/data/cylinder_data/sol-*.q`

The Hour 2 notebook (`01_data_loading.ipynb`) assumes these files live here and loads them via relative paths like:

- `data/cylinder/cylinder.sp.x`
- `data/cylinder/sol-0000010.q`

If you have access to the **full** LES sequence, you can point the notebook to a directory with many more `sol-*.q` files; the code will work the same way, just with more frames.

