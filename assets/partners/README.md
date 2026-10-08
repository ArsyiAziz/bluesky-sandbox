# Partner logos

The designer's About dialog shows these logos, read from this folder at build
time (`npm run build` in `src/bluesky_sandbox/ui/designer/web`). Each is named
by its key; `<key>-on-dark` is the version for the dark theme.

| File                  | From                                                   | Card                                   |
|-----------------------|--------------------------------------------------------|----------------------------------------|
| `gwu.png`             | `source/gwu-primary-logos.zip`: `gw_primary_2c.png`    | IASL, at GWU                           |
| `gwu-on-dark.png`     | `source/gwu-primary-logos.zip`: `gw_primary_2c_rev.png`| IASL, at GWU                           |
| `tudelft.svg`         | `source/tudelft-logos.zip`: `rgb/SVG/logo_rgb.svg`       | Operations and Environment Section |
| `tudelft-on-dark.svg` | `source/tudelft-logos.zip`: `rgb/SVG/logo_white_rgb.svg` | Operations and Environment Section |

The plain "TU Delft" logo, not the international one: that one's "Delft
University of Technology" is too small to read at the card's size, and the
card names the university in text.

The TU Delft files' canvas is tightened from `0 0 140 79` to `20 16 104 47`:
flush with the artwork on the left, as the GW logo is, and 4 units elsewhere
rather than the export's 20, since the card's own padding gives the logo its
clear space. The artwork is unchanged.

An `iasl.svg` (or `.png`) here would show beside GWU's on the IASL card.

`source/` holds the logo packages as received. Use the logos as each
institution's brand guidelines allow.
