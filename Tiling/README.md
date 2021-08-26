# Tile Generation from QuPath Annotated Images

This project starts with NDPI images of tissue and regions of diseased and
healthy tissue annotated using the QuPath system. The images are tiled and
each tile is classified based on its overlap of a region (different types
of diseased and health tissue).

1. Open qpproj project file in QuPath and check file location URIs.

Chances are that the paths to the images need to be changed to their current
location. QuPath usually finds them and suggests the appropriate changes.
But if not, then you'll have to find them by double-clicking on the old paths.
When done, quit QuPath.

2. Extract regions from QuPath into GeoJSON format.

Run QuPath Groovy script (script1.groovy) from the command-line on the
qpproj project file containing the annotations. For example, on a Mac,
this command looks like:

    /Applications/QuPath-0.2.3.app/Contents/MacOS/QuPath-0.2.3 script --project=/path/to/qupath/project/project.qpproj script1.groovy

The JSON files will be written to the current directory. Move them to the
directory where the NDPI images reside.

3. Extract TIFF images from NDPI images.

Use the ndpi2tiff tool available from
https://www.imnc.in2p3.fr/pagesperso/deroulers/software/ndpitools/. These NDPI
images contain multiple versions of the image with different sizes. This first
image is the largest, so we extract just that one using the ",0" suffix. So,
the command looks like:

    ndpi2tiff <image>,0

4. Generate tiles from TIFF images overlapping JSON regions.

Run the generatetiles.py script to generate tiles from the TIFF images
according to the regions in the JSON file. See comments at top of script file
for details. The typical command is:

    python3 generatetiles.py --other '<image>.tif' '<image>.ndpi.annotations.json'

Usage: `python3 generatetiles2.py [--other] <image> <annotations>`

The program generates tiles from the given `<image>` according to the given
`<annotations>`. The `<annotations>` file is a GeoJSON-formatted array from
the QuPath program. Each annotation includes "geometry":"coordinates" of the
points of a polygon encompassing a region, and "properties":"classification":"name"
of the pathology of the region. If a tile overlaps one region, but no others,
then it is written to the directory named after that region's pathology. The amount
of overlap necessary is controlled by the `TILE_OVERLAP` variable.

The tile images are stored in the `tiles/<pathology>` subdirectory. The tile image
file name is of the form: `<image>_<NNNNN>.jpg`, where `<NNNNN>` is a unique 5-digit,
0-padded number assigned to the tile image. The details about the tiles are
appended to the file `tiles/tiles.csv` (image, location, pathology, color).

If the optional `--other` argument is given, then the program also generates tiles
that are nearby, but don't overlap, the annotation regions. The tiles are designated
with pathology `OTHER\_PATHOLOGY` and color `OTHER\_COLOR` (set in code).

Finally, if the global variable `gGenerateTiledImage=True`, the program generates
the image `<image>_tiles.tif` that shows all the generated tiles as rectangles
colored according to their pathology.

NOTE: This program does not remove existing tiles, will overwrite
existing tiles with the same name, and will append tile information to
an existing `tiles/tiles.csv` file. This allows you to run the program
multiple times, once for each image, and collect all the tiles in one place.

## Author

Larry Holder, Washington State University


