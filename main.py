from pathlib import Path
import uuid
import numpy as np
import pandas as pd
import plotly.express as px


def latlon_to_xyz(lat, lon, radius_km=6371.0, normalize: bool = False):
    """Convert latitude and longitude (degrees) to ECEF x,y,z in kilometers.

    lat, lon may be scalars or array-like (degrees). Returns (x, y, z) in km
    with origin at Earth's center and assuming a spherical Earth of radius
    `radius_km`.

    If `normalize=True`, returns unit vectors (x,y,z) normalized to length 1
    rather than scaled by `radius_km`.
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    cos_lat = np.cos(lat_rad)
    x = radius_km * cos_lat * np.cos(lon_rad)
    y = radius_km * cos_lat * np.sin(lon_rad)
    z = radius_km * np.sin(lat_rad)

    if normalize:
        # compute norms and avoid division by zero
        norms = np.sqrt(x * x + y * y + z * z)
        # where norms == 0 keep zeros (shouldn't happen for valid lat/lon)
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.where(norms != 0, x / norms, 0.0)
            y = np.where(norms != 0, y / norms, 0.0)
            z = np.where(norms != 0, z / norms, 0.0)

    return x, y, z


def create_dummy_dataset(n=1000, dim=128, out_path="data/dummy_embeddings.csv", normalize: bool = False):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    embeddings = np.random.randn(n, dim).astype(np.float32)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    lat = np.random.uniform(-90.0, 90.0, size=n)
    lon = np.random.uniform(-180.0, 180.0, size=n)

    df = pd.DataFrame({
        "id": ids,
        "lat": lat,
        "lon": lon,
    })

    # add Cartesian coordinates (km) using spherical Earth approximation
    x, y, z = latlon_to_xyz(lat, lon, normalize=normalize)
    df["x_km"] = x
    df["y_km"] = y
    df["z_km"] = z

    if normalize:
        # normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
    for i in range(dim):
        df[f"emb_{i}"] = embeddings[:, i]

    df.to_csv(out, index=False)
    print(f"Wrote {n} rows with dim={dim} to {out}")


def get_top_k_similar(index: int, top_k: int = 20, use_cartesian: bool = False, path: str = "data/dummy_embeddings.csv"):
    """Return the top_k most similar row indices to the row at `index`.

    Similarity is cosine similarity on the embedding vector. If `use_cartesian`
    is True, the row's Cartesian coordinates (`x_km`,`y_km`,`z_km`) are used in
    addition to the embedding. If the CSV does not contain Cartesian columns,
    they will be computed from `lat`/`lon`.

    Returns a list of integer row indices (0-based) of length <= top_k.
    """
    df = pd.read_csv(path)

    # collect embedding columns in order
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    emb_cols = sorted(emb_cols, key=lambda s: int(s.split("_")[1]))
    if len(emb_cols) == 0:
        raise ValueError("No embedding columns found in csv")

    embeddings = df[emb_cols].to_numpy(dtype=np.float32)

    if use_cartesian:
        if {"x_km", "y_km", "z_km"}.issubset(df.columns):
            coords = df[["x_km", "y_km", "z_km"]].to_numpy(dtype=np.float32)
        else:
            coords = np.stack(latlon_to_xyz(df["lat"].to_numpy(), df["lon"].to_numpy(), normalize=False), axis=1).astype(np.float32)

        # standardize coords to zero mean unit variance so scale is comparable
        means = coords.mean(axis=0, keepdims=True)
        stds = coords.std(axis=0, keepdims=True)
        coords_scaled = (coords - means) / (stds + 1e-12) * 10.0  # scale factor to adjust importance of coords

        vectors = np.hstack([embeddings, coords_scaled])
    else:
        vectors = embeddings

    # normalize vectors to unit length for cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / (norms + 1e-12)

    n = vectors.shape[0]
    if not (0 <= index < n):
        raise IndexError(f"index {index} out of range for {n} rows")

    query = vectors[index]
    sims = vectors.dot(query)
    sims[index] = -np.inf

    k = min(top_k, n - 1)
    top_idx = np.argpartition(-sims, k - 1)[:k]
    # sort the top k indices by similarity descending
    top_idx = top_idx[np.argsort(-sims[top_idx])]

    top_list = top_idx.tolist()

    # instead of returning indices, plot them (target highlighted)
    plot_similar_locations(top_list, index, path=path, top_k=k, use_cartesian=use_cartesian)

    return None


def plot_similar_locations(top_indices, target_index, path: str = "data/dummy_embeddings.csv", title: str | None = None, top_k: int | None = None, use_cartesian: bool = False, filename: str | None = None):
    """Plot the lat/lon of `top_indices` and the `target_index` on a world map.

    - `top_indices`: sequence of integer row indices (0-based) to plot as neighbors
    - `target_index`: integer row index for the query point (plotted in a distinct color)
    - `path`: path to the CSV containing `lat`/`lon` and optional `id` columns

    This uses Plotly Express `scatter_geo` to produce an interactive map and
    calls `fig.show()` to render it in a browser or supported frontend.
    """
    df = pd.read_csv(path)

    if not (0 <= target_index < len(df)):
        raise IndexError(f"target_index {target_index} out of range for {len(df)} rows")

    neigh = df.iloc[top_indices].copy()
    neigh = neigh.reset_index(drop=True)
    neigh["type"] = "neighbor"

    targ = df.iloc[[target_index]].copy()
    targ = targ.reset_index(drop=True)
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

    # create output directory and derive filename if not provided
    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    k = top_k if top_k is not None else len(top_indices)
    filename = filename or f"similar_target{target_index}_k{k}_cart{int(bool(use_cartesian))}.html"
    out_path = out_dir / filename

    fig.write_html(str(out_path))
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    create_dummy_dataset(n=1000, dim=128)
