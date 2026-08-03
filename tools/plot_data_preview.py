from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import rasterio
from rasterio.warp import transform


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATE = "20180101"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def first_three_columns(path):
    """Read station name and two coordinate columns from an xlsx without pandas."""
    with ZipFile(path) as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ET.fromstring(book.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.text or "" for node in item.findall(".//m:t", NS))
                for item in root.findall("m:si", NS)
            ]
        sheet = ET.fromstring(book.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet.findall(".//m:sheetData/m:row", NS)[1:]:
            values = {}
            for cell in row.findall("m:c", NS):
                column = "".join(ch for ch in cell.get("r", "") if ch.isalpha())
                if column not in {"A", "B", "C"}:
                    continue
                value_node = cell.find("m:v", NS)
                if value_node is None:
                    continue
                value = value_node.text
                if cell.get("t") == "s":
                    value = strings[int(value)]
                values[column] = value
            try:
                first, second = float(values["B"]), float(values["C"])
            except (KeyError, TypeError, ValueError):
                continue
            # These workbooks have latitude under "longitude" and vice versa.
            lon, lat = (second, first) if second < -50 and 20 <= first <= 60 else (first, second)
            # Raster covers the contiguous United States; discard missing/malformed coordinates.
            if -125 <= lon <= -66 and 24 <= lat <= 50:
                rows.append((values.get("A", ""), lon, lat))
        return rows


def main():
    tif = DATA / "usa_smt-1" / f"sm_{DATE}_new.tif"
    with rasterio.open(tif) as src:
        image = src.read(1, masked=True)
        bounds = src.bounds
        raster_info = f"CRS: {src.crs}; size: {src.width} x {src.height}"

    groups = {}
    for path in sorted((DATA / "2018").glob("2018site_A_*.xlsx")):
        name = path.stem.removeprefix("2018site_A_")
        groups[name] = first_three_columns(path)

    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    finite = image.compressed()
    vmin, vmax = np.nanpercentile(finite, [2, 98])
    layer = ax.imshow(
        image,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
        cmap="YlGnBu",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    styles = ["o", "^", "s", "D"]
    colors = ["#e63946", "#ff9f1c", "#8338ec", "#00b4d8"]
    handles = []
    total = 0
    for (name, sites), marker, color in zip(groups.items(), styles, colors):
        lon = [item[1] for item in sites]
        lat = [item[2] for item in sites]
        lon, lat = transform("EPSG:4326", src.crs, lon, lat)
        total += len(sites)
        ax.scatter(lon, lat, s=30, marker=marker, c=color, edgecolors="white", linewidths=0.55, alpha=0.9)
        handles.append(Line2D([], [], marker=marker, linestyle="", markerfacecolor=color,
                              markeredgecolor="white", markersize=8, label=f"{name} ({len(sites)})"))

    colorbar = fig.colorbar(layer, ax=ax, shrink=0.86, pad=0.02)
    colorbar.set_label("TIF pixel value")
    ax.legend(handles=handles, title=f"Station networks (total {total})", loc="lower left",
              framealpha=0.92, fontsize=9)
    ax.set(title=f"Soil moisture raster and stations — {DATE[:4]}-{DATE[4:6]}-{DATE[6:]}",
           xlabel="Easting (m, EPSG:6933)", ylabel="Northing (m, EPSG:6933)")
    ax.grid(color="white", alpha=0.22, linewidth=0.6)
    ax.text(0.995, 0.01, raster_info, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="0.2", bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none"})
    output = ROOT / "tmp" / "data_preview_20180101.png"
    fig.savefig(output, dpi=180)
    print(output)
    print({name: len(sites) for name, sites in groups.items()})


if __name__ == "__main__":
    main()
