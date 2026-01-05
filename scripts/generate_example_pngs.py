from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px


def latlon_to_xyz(lat, lon, radius_km=6371.0):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cos_lat = np.cos(lat_rad)
    x = radius_km * cos_lat * np.cos(lon_rad)
    y = radius_km * cos_lat * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)
    return x, y, z


def top_k_indices(df, index, top_k=20, use_cartesian=False):
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb_cols = sorted(emb_cols, key=lambda s: int(s.split("_")[1]))
    embeddings = df[emb_cols].to_numpy(dtype=np.float32)

    if use_cartesian:
        if {"x_km", "y_km", "z_km"}.issubset(df.columns):
            coords = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float32)
        else:
            coords = np.stack(latlon_to_xyz(df["lat"].to_numpy(), df["lon"].to_numpy()), axis=1).astype(np.float32)
        means = coords.mean(axis=0, keepdims=True)
        stds = coords.std(axis=0, keepdims=True)
        coords_scaled = (coords - means) / (stds + 1e-12) * 10.0
        vectors = np.hstack([embeddings, coords_scaled])
    else:
        vectors = embeddings

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-12)

    query = vectors[index]
    sims = vectors.dot(query)
    sims[index] = -np.inf
    k = min(top_k, len(df) - 1)
    top_idx = np.argpartition(-sims, k - 1)[:k]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx.tolist()


def plot_and_save(df, top_indices, target_index, out_path_html, out_path_png, title=None):
    neigh = df.iloc[top_indices].reset_index(drop=True).copy()
    neigh["type"] = "neighbor"
    targ = df.iloc[[target_index]].reset_index(drop=True).copy()
    targ["type"] = "target"
    plot_df = pd.concat([neigh, targ], ignore_index=True)
    title = title or f"Top {len(top_indices)} similar locations to index {target_index}"

    fig = px.scatter_geo(
        plot_df,
        lat="lat",
        lon="lon",
        color="type",
        hover_name="id" if "id" in plot_df.columns else None,
        scope="world",
        title=title,
        category_orders={"type": ["neighbor", "target"]},
    )
    fig.update_traces(marker=dict(size=8))
    fig.write_html(str(out_path_html))
    # write PNG via kaleido
    fig.write_image(str(out_path_png), engine="kaleido")


def main():
    path = Path("data/dummy_embeddings.csv")
    df = pd.read_csv(path)
    target = 0
    top_k = 20

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # without Cartesian
    top_no_geo = top_k_indices(df, target, top_k=top_k, use_cartesian=False)
    out_html_no = out_dir / f"similar_target{target}_k{top_k}_cart0.html"
    out_png_no = out_dir / f"similar_target{target}_k{top_k}_cart0.png"
    plot_and_save(df, top_no_geo, target, out_html_no, out_png_no, title="No Cartesian (embeddings only)")

    # with Cartesian
    top_geo = top_k_indices(df, target, top_k=top_k, use_cartesian=True)
    out_html_geo = out_dir / f"similar_target{target}_k{top_k}_cart1.html"
    out_png_geo = out_dir / f"similar_target{target}_k{top_k}_cart1.png"
    plot_and_save(df, top_geo, target, out_html_geo, out_png_geo, title="With Cartesian appended")

    print("Wrote:")
    print(" ", out_html_no)
    print(" ", out_png_no)
    print(" ", out_html_geo)
    print(" ", out_png_geo)


if __name__ == "__main__":
    main()
