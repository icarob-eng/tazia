# Galaxy classification from galaxy zoo

From appendix A of https://arxiv.org/pdf/1308.3496.

- `A`: Artifact (or star);
- `E`: smooth (Elliptic);
  - `r`: round;
  - `i`: in between;
  - `c`: cigar shaped;
- `S`: Spiral;
  - `Se`: edge on;
    - `r`: round bulge;
    - `b`: boxy bulge;
    - `n`: no bulge;
  - `SB`: Spiral Barred (oblique view, not edge on);
  - For both barred and non-barred, but not edge on:
    - `d`: no bulge;
    - `c`: noticeable bulge;
    - `b`: obvious bulge;
    - `a`: dominant bulge;
  - For both edge-on and barred spirals, follows:
    - Number of bars (`1`,`2`,`3`,`4`,`+`,`?`);
    - Winding:
      - `t`: tight;
      - `m`: medium;
      - `l`: loose;
- For all galaxies, "odd" appearances:
  - `(r)`: ring;
  - `(l)`: lens/arc;
  - `(d)`: disturbed;
  - `(i)`: irregular;
  - `(o)`: other;
  - `(m)`: merger;
  - `(d)`: dust lane;


## Discards
In order to simplify the classification algorithm, we opted to discard,
firstly, any artifacts (`A`) and any "odd" appearances (`(*)`). We will
also remove galaxies witch are edge on (`Se`) or cigar-chapped (`Ec`). 
Galaxies with dominant bulge (`Sa` or `SBa`) are also discarded since
the bulge could look like a spiral galaxy.